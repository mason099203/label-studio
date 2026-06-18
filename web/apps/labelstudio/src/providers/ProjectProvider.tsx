import type React from "react";
import { createContext, useCallback, useContext, useEffect, useState } from "react";
import { shallowEqualObjects } from "shallow-equal";
import { addVisitedProject } from "@humansignal/core";
import { useAuth } from "@humansignal/core/providers/AuthProvider";
import { FF_UNSAVED_CHANGES, isFF } from "../utils/feature-flags";
import { useAPI, type WrappedResponse } from "./ApiProvider";
import { useAppStore } from "./AppStoreProvider";
import { useParams } from "./RoutesProvider";
import { atom, useSetAtom } from "jotai";

type Empty = Record<string, never>;

export const projectAtom = atom<APIProject | Empty>({});

type Context = {
  project: APIProject | Empty;
  fetchProject: (id?: string | number, force?: boolean) => Promise<APIProject | void>;
  updateProject: (fields: APIProject) => Promise<WrappedResponse<APIProject>>;
  invalidateCache: () => void;
};

export const ProjectContext = createContext<Context>({} as Context);
ProjectContext.displayName = "ProjectContext";

const projectCache = new Map<number, APIProject>();

type UpdateProjectOptions = {
  returnErrors?: boolean;
};

export const ProjectProvider: React.FunctionComponent = ({ children }) => {
  const api = useAPI();
  const params = useParams();
  const projectId = params.id;
  const { user } = useAuth();
  const { update: updateStore } = useAppStore();
  const [projectData, _setProjectData] = useState<APIProject | Empty>(
    () => projectCache.get(+projectId) ?? {},
  );
  const setProject = useSetAtom(projectAtom);

  const setProjectData = (project: APIProject | Empty) => {
    _setProjectData(project);
    setProject(project);
  };

  const fetchProject: Context["fetchProject"] = useCallback(
    async (id, force = false) => {
      const finalProjectId = +(id ?? projectId);

      if (isNaN(finalProjectId)) return;

      if (!force && projectCache.has(finalProjectId)) {
        const cached = projectCache.get(finalProjectId)!;
        setProjectData(cached);
        return cached;
      }

      const result = await api.callApi<APIProject>("project", {
        params: { pk: finalProjectId },
        errorFilter: () => false,
      });

      const projectInfo = result as unknown as APIProject;
      if (!projectInfo?.id) return;

      const existing = projectCache.get(finalProjectId);
      if (existing && shallowEqualObjects(existing, projectInfo)) {
        return existing;
      }

      projectCache.set(projectInfo.id, projectInfo);
      setProjectData(projectInfo);
      updateStore({ project: projectInfo });

      if (projectInfo.id) {
        addVisitedProject(projectInfo.id, user?.id);
      }

      return projectInfo;
    },
    [projectId, api, updateStore, user?.id],
  );

  const updateProject: Context["updateProject"] = useCallback(
    async (fields: APIProject, options?: UpdateProjectOptions) => {
      const result = await api.callApi<APIProject>("updateProject", {
        params: {
          pk: projectData.id,
        },
        body: fields,
        errorFilter: options?.returnErrors ? undefined : () => true,
      });

      if (isFF(FF_UNSAVED_CHANGES)) {
        if (result?.$meta?.ok) {
          setProjectData(result as unknown as APIProject);
          updateStore({ project: result });
          projectCache.set(result.id, result);
        }
      } else {
        if (result.$meta) {
          setProjectData(result as unknown as APIProject);
          updateStore({ project: result });
        }
      }

      return result;
    },
    [projectData, api, updateStore],
  );

  useEffect(() => {
    const id = +projectId;
    if (isNaN(id)) return;
    fetchProject(id);
  }, [projectId, fetchProject]);

  useEffect(() => {
    return () => projectCache.clear();
  }, []);

  return (
    <ProjectContext.Provider
      value={{
        project: projectData,
        fetchProject,
        updateProject,
        invalidateCache() {
          projectCache.clear();
          setProjectData({});
        },
      }}
    >
      {children}
    </ProjectContext.Provider>
  );
};

export const useProject: () => Context = () => {
  return useContext(ProjectContext) ?? {};
};
