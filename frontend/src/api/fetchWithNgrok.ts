/**
 * Thin wrapper around fetch() that:
 * 1. Automatically prepends VITE_API_BASE to relative /api/* paths
 * 2. Adds the `ngrok-skip-browser-warning` header so the ngrok
 *    interstitial page is bypassed in production
 *
 * Safe to use in all environments — both headers/rewrites are harmless
 * when not behind ngrok.
 */
const API_BASE = ((import.meta.env.VITE_API_BASE as string) || '').replace(/\/$/, '')

export function fetchApi(url: string, init: RequestInit = {}): Promise<Response> {
  // Prepend API_BASE for relative paths (e.g. /api/workspaces)
  const fullUrl = url.startsWith('/') ? `${API_BASE}${url}` : url

  const headers = new Headers(init.headers)
  headers.set('ngrok-skip-browser-warning', 'true')
  return fetch(fullUrl, { ...init, headers })
}
