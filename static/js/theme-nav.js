/**
 * NeuroMed AI — Clinical Decision Support System (CDSS)
 * Global Theme Switcher & Enterprise Navigation Controller
 */

function toggleTheme() {
  const current = document.documentElement.getAttribute('data-theme') || 'light';
  const next = current === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  try {
    localStorage.setItem('neuromed_theme', next);
  } catch (e) {
    console.warn('LocalStorage unavailable for theme persistence', e);
  }
  updateThemeIcon(next);
}

function updateThemeIcon(theme) {
  const icon = document.getElementById('themeIcon');
  if (icon) {
    icon.setAttribute('data-lucide', theme === 'dark' ? 'sun' : 'moon');
    if (typeof lucide !== 'undefined') {
      lucide.createIcons();
    }
  }
}

function toggleNavMenu() {
  const links = document.getElementById('main-nav-links');
  if (links) {
    links.classList.toggle('nav-open');
  }
}

// Auto-initialize theme icon and icons on load
document.addEventListener('DOMContentLoaded', () => {
  const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
  updateThemeIcon(currentTheme);
  if (typeof lucide !== 'undefined') {
    lucide.createIcons();
  }
});

