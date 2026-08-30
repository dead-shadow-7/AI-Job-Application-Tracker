import { Navigate, Route, Routes } from 'react-router-dom'
import { ProtectedRoute } from '@/auth/ProtectedRoute'
import { AppShell } from '@/components/AppShell'
import { ApplicationDetail } from '@/routes/ApplicationDetail'
import { Applications } from '@/routes/Applications'
import { Login } from '@/routes/Login'
import { NewApplication } from '@/routes/NewApplication'
import { PasteJob } from '@/routes/PasteJob'

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route element={<ProtectedRoute />}>
        <Route element={<AppShell />}>
          <Route path="/" element={<Applications />} />
          <Route path="/applications/paste" element={<PasteJob />} />
          <Route path="/applications/new" element={<NewApplication />} />
          <Route path="/applications/:id" element={<ApplicationDetail />} />
        </Route>
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
