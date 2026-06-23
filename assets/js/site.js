// Theme toggle
const html = document.documentElement;
const themeToggle = document.getElementById('themeToggle');

function setTheme(theme) {
    html.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
    const icon = themeToggle.querySelector('i');
    icon.className = theme === 'light' ? 'fa-solid fa-moon' : 'fa-solid fa-sun';
}

// Restore saved theme, default to dark
setTheme(localStorage.getItem('theme') || 'dark');

themeToggle.addEventListener('click', () => {
    setTheme(html.getAttribute('data-theme') === 'light' ? 'dark' : 'light');
});

// Scroll fade-in animations
const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) entry.target.classList.add('visible');
    });
}, { threshold: 0.1 });

document.querySelectorAll('.fade-in').forEach(el => observer.observe(el));
