import { Outlet, Route, Routes } from 'react-router-dom';

import { AppHeader, Page, StatusMessage } from './components';
import { AnalyzingPage } from './pages/AnalyzingPage';
import { BreakdownView } from './pages/BreakdownPage';
import { HomePage } from './pages/HomePage';
import { LiveCameraPage } from './pages/LiveCameraPage';
import { ResultsLayout, ResultVideoView } from './pages/ResultsPage';

export function App() {
  return (
    <Routes>
      <Route
        element={
          <>
            <AppHeader />
            <Outlet />
          </>
        }
      >
        <Route index element={<HomePage />} />
        <Route path="live" element={<LiveCameraPage />} />
        <Route path="analyzing/:video" element={<AnalyzingPage />} />
        <Route path="results/:video" element={<ResultsLayout />}>
          <Route index element={<ResultVideoView />} />
          <Route path="breakdown" element={<BreakdownView />} />
        </Route>
        <Route
          path="*"
          element={
            <Page>
              <StatusMessage title="Page not found." showHomeLink />
            </Page>
          }
        />
      </Route>
    </Routes>
  );
}
