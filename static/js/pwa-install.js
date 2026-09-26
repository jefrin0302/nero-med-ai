// pwa-install.js — Production PWA Registration & Install Handler
let deferredPrompt = null;

// 1. Register Service Worker on Root Scope
if ("serviceWorker" in navigator) {
    window.addEventListener("load", () => {
        navigator.serviceWorker.register("/service-worker.js")
            .then(reg => {
                console.log("[PWA] Service Worker active with scope:", reg.scope);
            })
            .catch(err => {
                console.warn("[PWA] Service Worker registration failed:", err);
            });
    });
}

// 2. Capture Browser Install Prompt
window.addEventListener("beforeinstallprompt", (e) => {
    e.preventDefault();
    deferredPrompt = e;

    // Reveal Install Buttons across Navigation
    const installBtns = document.querySelectorAll(".pwa-install-btn");
    installBtns.forEach(btn => {
        btn.style.display = "inline-flex";
    });

    console.log("[PWA] Install prompt captured and ready");
});

// 3. User Triggered Install Action
function triggerPWAInstall() {
    if (deferredPrompt) {
        deferredPrompt.prompt();
        deferredPrompt.userChoice.then((choiceResult) => {
            if (choiceResult.outcome === "accepted") {
                console.log("[PWA] User accepted the installation");
            } else {
                console.log("[PWA] User dismissed the installation");
            }
            deferredPrompt = null;
            hideInstallButtons();
        });
    } else {
        // Fallback instructions for iOS Safari or already installed browsers
        const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent) && !window.MSStream;
        if (isIOS) {
            alert("To install Wilson Disease AI on your iPhone / iPad:\n\n1. Tap the Share button at the bottom of Safari (square with arrow)\n2. Scroll down and tap 'Add to Home Screen'\n3. Tap 'Add' in the top right corner.");
        } else {
            alert("Wilson AI PWA Installation:\n\nIf you are on desktop Chrome or Edge, click the Install icon (computer screen with down-arrow) in the right side of the address bar.\n\nOn Android, tap the browser menu (⋮) -> 'Install App' or 'Add to Home screen'.");
        }
    }
}

// 4. Handle Successful Installation
window.addEventListener("appinstalled", () => {
    console.log("[PWA] App successfully installed to device!");
    deferredPrompt = null;
    hideInstallButtons();
});

function hideInstallButtons() {
    const installBtns = document.querySelectorAll(".pwa-install-btn");
    installBtns.forEach(btn => {
        btn.style.display = "none";
    });
}
