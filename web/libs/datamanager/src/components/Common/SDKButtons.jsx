import { useSDK } from "../../providers/SDKProvider";
import { Button } from "@humansignal/ui";

const SDKButton = ({ eventName, testId, ...props }) => {
  const sdk = useSDK();

  return sdk.hasHandler(eventName) ? (
    <Button
      {...props}
      size={props.size ?? "small"}
      look={props.look ?? "outlined"}
      variant={props.variant ?? "neutral"}
      aria-label={`${eventName.replace("Clicked", "")} button`}
      data-testid={testId}
      onClick={() => {
        sdk.invoke(eventName);
      }}
    />
  ) : null;
};

export const SettingsButton = ({ ...props }) => {
  return <SDKButton {...props} eventName="settingsClicked" />;
};

export const ImportButton = ({ ...props }) => {
  return <SDKButton {...props} eventName="importClicked" testId="dm-import-button" />;
};

export const ExportButton = ({ ...props }) => {
  return <SDKButton {...props} eventName="exportClicked" testId="dm-export-button" />;
};

/**
 * Training 按鈕：觸發 trainingClicked 事件，導向訓練模組頁面。
 * @param {Object} props - 傳給 SDKButton 的屬性
 */
export const TrainingButton = ({ ...props }) => {
  return <SDKButton {...props} eventName="trainingClicked" testId="dm-training-button" />;
};
