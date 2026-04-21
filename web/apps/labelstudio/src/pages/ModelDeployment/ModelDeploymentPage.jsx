import { Tabs, TabsList, TabsTrigger, TabsContent } from "@humansignal/ui";
import { useUpdatePageTitle } from "@humansignal/core";
import { PlaygroundTab } from "./PlaygroundTab";
import { LogsAndMetricsTab } from "./LogsAndMetricsTab";
import { UploadTab } from "./UploadTab";
import { cn } from "../../utils/bem";
import "./ModelDeployment.scss";

const rootClass = cn("model-deployment-page");

/**
 * 模型部署頁面：提供「部署模型」、「模型測試」、「儀錶板」三個分頁
 * - 部署模型：透過 Triton Upload Server REST API 直接推送模型至推論倉庫
 * - 模型測試（Playground）：輸入 Triton URL + Key，對影像做推論測試
 * - 儀錶板：連結 Grafana / Prometheus / Triton metrics
 */
export function ModelDeploymentPage() {
  useUpdatePageTitle("模型部署");

  return (
    <div className={rootClass.toClassName()}>
      <Tabs defaultValue="playground" variant="flat" className={rootClass.elem("tabs").toClassName()}>
        <TabsList>
          <TabsTrigger value="deploy">部署模型</TabsTrigger>
          <TabsTrigger value="playground">模型測試</TabsTrigger>
          <TabsTrigger value="logs">儀錶板</TabsTrigger>
        </TabsList>
        <TabsContent value="deploy" className={rootClass.elem("tab-content").toClassName()}>
          <UploadTab />
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
