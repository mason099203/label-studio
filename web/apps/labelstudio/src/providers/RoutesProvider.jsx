import { createContext, useContext, useEffect, useMemo, useState } from "react";
import { generatePath, matchPath, useHistory, useLocation } from "react-router";
import { Pages } from "../pages";
import { setBreadcrumbs, useBreadcrumbControls } from "../services/breadrumbs";
import { pageSetToRoutes } from "../utils/routeHelpers";
import { useAppStore } from "./AppStoreProvider";
import { useConfig } from "./ConfigProvider";

export const RoutesContext = createContext();

const findMacthingComponents = (path, routesMap, parentPath = "") => {
  const result = [];

  if (path === "/") {
    const root = routesMap.at(0);
    if (root) {
      result.push({ ...root, path: root.path });
      if (root.routes) {
        result.push(...findMacthingComponents(path, root.routes, root.path));
      }
    }
    return result;
  }

  const candidates = routesMap
    .map((route) => {
      const isRoot = route.path === "/";
      const matchingPath = `${parentPath}${route.path}`;
      // Match prefix like react-router parent routes; only the root path uses exact.
      const segment = matchPath(path, { path: matchingPath, exact: isRoot });
      return segment ? { route, routePath: matchingPath } : null;
    })
    .filter(Boolean);

  // Prefer the most specific (longest) matching route at this level.
  candidates.sort((a, b) => b.routePath.length - a.routePath.length);

  const best = candidates[0];
  if (best) {
    result.push({ ...best.route, path: best.routePath });
    if (best.route.routes) {
      result.push(...findMacthingComponents(path, best.route.routes, best.routePath));
    }
  }

  return result;
};

export const RoutesProvider = ({ children }) => {
  const history = useHistory();
  const location = useFixedLocation();
  const config = useConfig();
  const { store } = useAppStore();
  const breadcrumbs = useBreadcrumbControls();
  const [currentContext, setCurrentContext] = useState(null);
  const [currentContextProps, setCurrentContextProps] = useState(null);

  const routesMap = useMemo(() => {
    return pageSetToRoutes(Pages, { config, store });
  }, [config, store]);

  const routesChain = useMemo(() => {
    return findMacthingComponents(location.pathname, routesMap);
  }, [location, routesMap]);

  const lastRoute = useMemo(() => {
    return routesChain.filter((r) => !r.modal).slice(-1)[0];
  }, [routesChain]);

  const [currentPath, setCurrentPath] = useState(lastRoute?.path);

  const contextValue = useMemo(
    () => ({
      routesMap,
      breadcrumbs,
      currentContext,
      setContextProps: setCurrentContextProps,
      path: currentPath,
      findComponent: (path) => findMacthingComponents(path, routesMap),
    }),
    [breadcrumbs, routesMap, currentContext, currentPath, setCurrentContext],
  );

  useEffect(() => {
    const ContextComponent = lastRoute?.context;

    setCurrentContext({
      component: ContextComponent ?? null,
      props: currentContextProps,
    });

    setCurrentPath(lastRoute?.path);

    try {
      const crumbs = routesChain
        .map((route) => {
          const params = matchPath(location.pathname, { path: route.path });
          const path = generatePath(route.path, params.params);
          const title = route.title instanceof Function ? route.title() : route.title;
          const key = route.component?.displayName ?? route.key ?? path;

          return { path, title, key };
        })
        .filter((c) => !!c.title);

      setBreadcrumbs(crumbs);
    } catch (err) {
      console.log(err);
    }
  }, [location, routesMap, currentContextProps, routesChain, lastRoute]);

  return <RoutesContext.Provider value={contextValue}>{children}</RoutesContext.Provider>;
};

export const useRoutesMap = () => {
  return useContext(RoutesContext)?.routesMap ?? [];
};

export const useFindRouteComponent = () => {
  return useContext(RoutesContext)?.findComponent ?? (() => null);
};

export const useBreadcrumbs = () => {
  return useBreadcrumbControls();
};

export const useCurrentPath = () => {
  return useContext(RoutesContext)?.path;
};

export const useParams = () => {
  const location = useFixedLocation();
  const currentPath = useCurrentPath();
  const routesMap = useRoutesMap();

  return useMemo(() => {
    const parsedLocation = location.search
      .replace(/^\?/, "")
      .split("&")
      .filter(Boolean)
      .map((pair) => {
        const [key, value] = pair.split("=").map((p) => decodeURIComponent(p));
        return [key, value];
      });

    const search = Object.fromEntries(parsedLocation);

    let params = { ...search };
    const chain = findMacthingComponents(location.pathname, routesMap);
    for (const route of chain) {
      const segment = matchPath(location.pathname, { path: route.path });
      if (segment?.params) {
        params = { ...params, ...segment.params };
      }
    }

    const urlParams = matchPath(location.pathname, currentPath ?? "");
    if (urlParams?.params) {
      params = { ...params, ...urlParams.params };
    }

    return params;
  }, [location.pathname, location.search, currentPath, routesMap]);
};

export const useContextComponent = () => {
  const ctx = useContext(RoutesContext);
  const { component: ContextComponent, props: contextProps } = ctx?.currentContext ?? {};

  return { ContextComponent, contextProps };
};

export const useFixedLocation = () => {
  const location = useLocation();

  location;

  const result = useMemo(() => {
    return location.location ?? location;
  }, [location]);

  return result;
};

export const useContextProps = () => {
  const setProps = useContext(RoutesContext).setContextProps;
  return useMemo(() => setProps, [setProps]);
};
