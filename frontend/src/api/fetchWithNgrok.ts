/**
 * Thin wrapper around fetch() that automatically adds the
 * `ngrok-skip-browser-warning` header so the ngrok interstitial page
 * is bypassed in production (backend tunnelled via ngrok).
 * Safe to use in all environments — the header is ignored by non-ngrok servers.
 */
export function fetchApi(url: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers)
  headers.set('ngrok-skip-browser-warning', 'true')
  return fetch(url, { ...init, headers })
}
