// VerifAI visual layer: animated molecular network (tsParticles), typed text (Typed.js),
// entrance + floating motion (GSAP) and animated counters (CountUp.js).
// Everything is opt-in through data attributes and is skipped when the user prefers reduced motion.
(function () {
  const reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ---------- molecular network background: <div data-vf-particles="light|dark"> ----------
  async function particles() {
    if (!window.tsParticles || reduced) return;
    if (window.loadSlim && !particles.loaded) { await window.loadSlim(window.tsParticles); particles.loaded = true; }
    const els = document.querySelectorAll('[data-vf-particles]');
    for (const el of els) {
      if (!el.id) el.id = 'vfp-' + Math.random().toString(36).slice(2, 8);
      const dark = el.dataset.vfParticles === 'dark';
      await window.tsParticles.load({
        id: el.id,
        options: {
          fullScreen: { enable: false },
          fpsLimit: 50,
          detectRetina: true,
          background: { color: { value: 'transparent' } },
          particles: {
            number: { value: dark ? 70 : 55, density: { enable: true, width: 900, height: 500 } },
            color: { value: dark ? ['#93C5FD', '#5EEAD4', '#FFFFFF'] : ['#2563EB', '#0D9488', '#60A5FA'] },
            links: { enable: true, distance: 130, color: dark ? '#93C5FD' : '#2563EB', opacity: dark ? 0.28 : 0.18, width: 1 },
            move: { enable: true, speed: 0.6, outModes: { default: 'bounce' } },
            opacity: { value: { min: 0.3, max: 0.8 } },
            size: { value: { min: 1.5, max: 3.5 } },
          },
          interactivity: {
            events: { onHover: { enable: true, mode: 'grab' }, resize: { enable: true } },
            modes: { grab: { distance: 160, links: { opacity: 0.55 } } },
          },
        },
      });
    }
  }

  // ---------- typed text: <span data-vf-typed='["one","two"]'> ----------
  function typed() {
    document.querySelectorAll('[data-vf-typed]').forEach(el => {
      let strings = [];
      try { strings = JSON.parse(el.dataset.vfTyped); } catch (e) { return; }
      if (!strings.length) return;
      if (!window.Typed || reduced) { el.textContent = strings[0]; return; }
      new window.Typed(el, { strings, typeSpeed: 32, backSpeed: 16, backDelay: 2200, loop: true, smartBackspace: true });
    });
  }

  // ---------- counters: <span data-vf-count="42" data-vf-decimals="1" data-vf-suffix="%"> ----------
  function counters() {
    const CU = window.countUp && window.countUp.CountUp;
    // Plain whole-number stat tiles (admin pages) count up too
    document.querySelectorAll('.stat-value').forEach(el => {
      const t = el.textContent.trim();
      if (/^\d{1,7}$/.test(t) && !el.querySelector('*')) el.dataset.vfCount = t;
    });
    document.querySelectorAll('[data-vf-count]').forEach(el => {
      const end = parseFloat(el.dataset.vfCount);
      if (!CU || reduced || !isFinite(end)) return;
      const c = new CU(el, end, {
        decimalPlaces: parseInt(el.dataset.vfDecimals || '0', 10),
        suffix: el.dataset.vfSuffix || '', duration: 1.4, useGrouping: true, enableScrollSpy: true, scrollSpyOnce: true,
      });
      if (!c.error) c.start();
    });
  }

  // ---------- motion: card entrance and floating chips ----------
  function motion() {
    if (!window.gsap || reduced) return;
    const gsap = window.gsap;
    const cards = document.querySelectorAll('.page-content .vf-card, .page-content .content-card, .page-content .stat-card, .vf-hero-body > *');
    if (cards.length) gsap.from(cards, { y: 14, opacity: 0, duration: 0.55, ease: 'power2.out', stagger: 0.045, clearProps: 'transform,opacity' });
    document.querySelectorAll('[data-vf-float] > *').forEach((chip, i) => {
      gsap.to(chip, { y: (i % 2 ? 10 : -10), x: (i % 3 - 1) * 6, duration: 3 + (i % 4) * 0.6,
                      ease: 'sine.inOut', yoyo: true, repeat: -1, delay: i * 0.25 });
    });
    document.querySelectorAll('[data-vf-pop]').forEach(el => {
      gsap.from(el, { scale: 0.92, opacity: 0, duration: 0.7, ease: 'back.out(1.6)', delay: 0.15 });
    });
  }

  function init() { particles(); typed(); counters(); motion(); }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
