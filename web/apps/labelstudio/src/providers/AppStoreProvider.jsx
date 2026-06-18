import React from "react";

const AppStoreContext = React.createContext();
AppStoreContext.displayName = "AppStoreContext";

export const AppStoreProvider = ({ children }) => {
  const [store, setStore] = React.useState({});

  const update = React.useCallback((newData) => {
    setStore((prev) => ({ ...prev, ...(newData ?? {}) }));
  }, []);

  const contextValue = React.useMemo(
    () => ({
      store,
      update,
    }),
    [store, update],
  );

  return <AppStoreContext.Provider value={contextValue}>{children}</AppStoreContext.Provider>;
};

export const useAppStore = () => {
  return React.useContext(AppStoreContext);
};
