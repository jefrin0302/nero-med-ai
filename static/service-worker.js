// service-worker.js — Wilson Disease AI Progressive Web App
const CACHE_NAME = "wilson-ai-v3";
const OFFLINE_URL = "/offline";

const STATIC_ASSETS = [
    "/",
    "/offline",
    "/manifest.json",
    "/static/manifest.json",
    "/static/css/modern_theme.css",
    "/static/icons/icon-192.png",
    "/static/icons/icon-512.png",
    "/static/js/pwa-install.js"
];

// 1. Install Event: Cache Core Static Shell
self.addEventListener("install", event => {
    event.waitUntil(
        caches.open(CACHE_NAME).then(cache => {
            console.log("[PWA ServiceWorker] Pre-caching static app shell");
            return cache.addAll(STATIC_ASSETS);
        }).then(() => self.skipWaiting())
    );
});

// 2. Activate Event: Clean up outdated caches
self.addEventListener("activate", event => {
    event.waitUntil(
        caches.keys().then(keyList => {
            return Promise.all(keyList.map(key => {
                if (key !== CACHE_NAME) {
                    console.log("[PWA ServiceWorker] Removing legacy cache:", key);
                    return caches.delete(key);
                }
            }));
        }).then(() => self.clients.claim())
    );
});

// 3. Fetch Event: Intelligent Strategy
self.addEventListener("fetch", event => {
    // Only handle GET requests
    if (event.request.method !== "GET") return;

    const url = new URL(event.request.url);

    // Bypass caching for dynamic prediction / RAG APIs
    if (url.pathname.startsWith("/api/") || url.pathname === "/submit") {
        return;
    }

    // Navigation Requests (HTML Pages): Network-First with Offline Fallback
    if (event.request.mode === "navigate") {
        event.respondWith(
            fetch(event.request)
                .then(networkResponse => {
                    // Update cache with fresh version
                    if (networkResponse && networkResponse.status === 200) {
                        const copy = networkResponse.clone();
                        caches.open(CACHE_NAME).then(cache => cache.put(event.request, copy));
                    }
                    return networkResponse;
                })
                .catch(async () => {
                    const cachedResponse = await caches.match(event.request);
                    if (cachedResponse) return cachedResponse;
                    return caches.match(OFFLINE_URL);
                })
        );
        return;
    }

    // Static Assets (CSS, JS, Images, Icons): Cache-First with Network Fallback
    event.respondWith(
        caches.match(event.request).then(cachedResponse => {
            if (cachedResponse) return cachedResponse;

            return fetch(event.request).then(networkResponse => {
                if (networkResponse && networkResponse.status === 200 && networkResponse.type === "basic") {
                    const copy = networkResponse.clone();
                    caches.open(CACHE_NAME).then(cache => cache.put(event.request, copy));
                }
                return networkResponse;
            }).catch(() => {
                // If offline and image requested, can return fallback if needed
            });
        })
    );
});
