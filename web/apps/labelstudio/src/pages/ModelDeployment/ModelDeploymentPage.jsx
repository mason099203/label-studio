import { Tabs, TabsList, TabsTrigger, TabsContent } from "@humansignal/ui";
import { useUpdatePageTitle } from "@humansignal/core";
import { DeployTab } from "./DeployTab";
import { PlaygroundTab } from "./PlaygroundTab";
import { LogsAndMetricsTab } from "./LogsAndMetricsTab";
import { cn } from "../../utils/bem";
import "./ModelDeployment.scss";

const rootClass = cn("model-deployment-page");

/**
 * 模型部署頁面：提供「模型部署」、「Playground」、「日誌與效能」三個分頁
 * - 模型部署：列出已部署模型、API 位址與需指定的 API Key
 * - Playground：輸入 URL + Key 測試 API
 * - 日誌與效能：連結至 Grafana / Prometheus / Triton metrics
 */
export function ModelDeploymentPage() {
  useUpdatePageTitle("模型部署");

  return (
    <div className={rootClass.toClassName()}>
      <Tabs defaultValue="deploy" variant="flat" className={rootClass.elem("tabs").toClassName()}>
        <TabsList>
          <TabsTrigger value="deploy">模型部署</TabsTrigger>
          <TabsTrigger value="playground">Playground</TabsTrigger>
          <TabsTrigger value="logs">日誌與效能</TabsTrigger>
        </TabsList>
        <TabsContent value="deploy" className={rootClass.elem("tab-content").toClassName()}>
          <DeployTab />
        </TabsContent>
        <TabsContent value="playground" className={rootClass.elem("tab-content").toClassName()}>
          <PlaygroundTab />
        </TabsContent>
        <TabsContent value="logs" className={rootClass.elem("tab-content").toClassName()}>
          <LogsAndMetricsTab />
        </TabsContent>
      </Tabs>
    </div>
  );
}

ModelDeploymentPage.title = "模型部署";
ModelDeploymentPage.titleRaw = "模型部署";
ModelDeploymentPage.path = "/model-deployment";
