import { Tabs, TabsList, TabsTrigger, TabsContent } from "@humansignal/ui";
import { useUpdatePageTitle } from "@humansignal/core";
import { PlaygroundTab } from "./PlaygroundTab";
import { LogsAndMetricsTab } from "./LogsAndMetricsTab";
import { cn } from "../../utils/bem";
import "./ModelDeployment.scss";

const rootClass = cn("model-deployment-page");

/**
 * 模型部署頁面：提供「模型測試」與「儀錶板」兩個分頁
 * - 模型測試（Playground）：輸入 Triton URL + Key，對影像做推論測試
 * - 儀錶板：查看部署模型列表、推論事件、效能快照與監控入口；支援移除部署模型
 */
export function ModelDeploymentPage() {
  useUpdatePageTitle("模型部署");

  return (
    <div className={rootClass.toClassName()}>
      <Tabs defaultValue="playground" variant="flat" className={rootClass.elem("tabs").toClassName()}>
        <TabsList>
          <TabsTrigger value="playground">模型測試</TabsTrigger>
          <TabsTrigger value="logs">儀錶板</TabsTrigger>
        </TabsList>
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
