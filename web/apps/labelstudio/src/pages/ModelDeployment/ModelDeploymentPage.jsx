import { lazy, Suspense, useCallback, useMemo } from "react";
import { useHistory } from "react-router";
import { Tabs, TabsList, TabsTrigger, TabsContent, Spinner } from "@humansignal/ui";
import { useUpdatePageTitle } from "@humansignal/core";
import { useFixedLocation } from "../../providers/RoutesProvider";
import { cn } from "../../utils/bem";
import "./ModelDeployment.scss";

const rootClass = cn("model-deployment-page");

const PlaygroundTab = lazy(() =>
  import("./PlaygroundTab").then((module) => ({ default: module.PlaygroundTab })),
);
const LogsAndMetricsTab = lazy(() =>
  import("./LogsAndMetricsTab").then((module) => ({ default: module.LogsAndMetricsTab })),
);

const TAB_QUERY_KEY = "tab";
const DEFAULT_TAB = "playground";

/** @type {Record<string, { value: string, label: string }>} */
export const MODEL_DEPLOYMENT_TABS = {
  playground: { value: "playground", label: "模型測試" },
  logs: { value: "logs", label: "儀錶板" },
};

const VALID_TAB_VALUES = new Set(Object.values(MODEL_DEPLOYMENT_TABS).map((tab) => tab.value));

/**
 * 從 URL query 解析目前應顯示的分頁。
 * @param {string} search
 * @returns {string}
 */
function parseTabFromSearch(search) {
  const params = new URLSearchParams(search.replace(/^\?/, ""));
  const tab = params.get(TAB_QUERY_KEY);
  return VALID_TAB_VALUES.has(tab) ? tab : DEFAULT_TAB;
}

/**
 * 模型部署頁面：提供「模型測試」與「儀錶板」兩個分頁
 * - 模型測試（Playground）：選擇專案與 Triton 位址，對影像做推論測試
 * - 儀錶板：查看部署模型列表、推論事件、效能快照與監控入口；支援移除部署模型
 *
 * 分頁狀態同步至 URL query（`?tab=playground` / `?tab=logs`），方便書籤與分享。
 */
export function ModelDeploymentPage() {
  useUpdatePageTitle("模型部署");

  const history = useHistory();
  const location = useFixedLocation();
  const activeTab = useMemo(() => parseTabFromSearch(location.search), [location.search]);

  const handleTabChange = useCallback(
    (nextTab) => {
      if (!VALID_TAB_VALUES.has(nextTab) || nextTab === activeTab) return;

      const params = new URLSearchParams(location.search.replace(/^\?/, ""));
      if (nextTab === DEFAULT_TAB) {
        params.delete(TAB_QUERY_KEY);
      } else {
        params.set(TAB_QUERY_KEY, nextTab);
      }

      const query = params.toString();
      history.replace({
        pathname: location.pathname,
        search: query ? `?${query}` : "",
      });
    },
    [activeTab, history, location.pathname, location.search],
  );

  return (
    <div className={rootClass.toClassName()}>
      <header className={rootClass.elem("header").toClassName()}>
        <h2 className={rootClass.elem("title").toClassName()}>模型部署</h2>
        <p className={rootClass.elem("subtitle").toClassName()}>
          測試已部署至 Triton 的模型，並監控推論效能與部署狀態。
        </p>
      </header>

      <Tabs
        value={activeTab}
        onValueChange={handleTabChange}
        variant="flat"
        className={rootClass.elem("tabs").toClassName()}
      >
        <TabsList>
          {Object.values(MODEL_DEPLOYMENT_TABS).map((tab) => (
            <TabsTrigger key={tab.value} value={tab.value}>
              {tab.label}
            </TabsTrigger>
          ))}
        </TabsList>

        <TabsContent value={MODEL_DEPLOYMENT_TABS.playground.value} className={rootClass.elem("tab-content").toClassName()}>
          <Suspense
            fallback={
              <div className={rootClass.elem("tab-fallback").toClassName()}>
                <Spinner size={32} />
              </div>
            }
          >
            <PlaygroundTab />
          </Suspense>
        </TabsContent>

        <TabsContent value={MODEL_DEPLOYMENT_TABS.logs.value} className={rootClass.elem("tab-content").toClassName()}>
          <Suspense
            fallback={
              <div className={rootClass.elem("tab-fallback").toClassName()}>
                <Spinner size={32} />
              </div>
            }
          >
            <LogsAndMetricsTab />
          </Suspense>
        </TabsContent>
      </Tabs>
    </div>
  );
}

ModelDeploymentPage.title = "模型部署";
ModelDeploymentPage.titleRaw = "模型部署";
ModelDeploymentPage.path = "/model-deployment";
