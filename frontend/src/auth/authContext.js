import { createContext, useContext } from 'react'

/**
 * Context and hook live apart from the provider component.
 *
 * React Fast Refresh only preserves state for modules that export components
 * exclusively; mixing a hook into AuthProvider.jsx makes every auth edit remount
 * the tree and drop the session mid-development.
 */
export const AuthContext = createContext(null)

export function useAuth() {
  const context = useContext(AuthContext)
  if (!context) throw new Error('useAuth must be used inside an AuthProvider')
  return context
}
