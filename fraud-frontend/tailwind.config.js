/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        ink: {
          DEFAULT: '#F8FAFC',
          surface: '#FFFFFF',
          raised: '#F1F5F9',
          border: '#E2E8F0',
        },
        slate: {
          muted: '#64748B',
          text: '#0F172A',
        },
        signal: {
          DEFAULT: '#B45309',
          dim: '#92400E',
        },
        risk: {
          DEFAULT: '#DC2626',
          dim: '#991B1B',
        },
        graph: {
          DEFAULT: '#0F766E',
          dim: '#134E4A',
        },
      },
      fontFamily: {
        display: ['"Sora"', 'sans-serif'],
        body: ['"IBM Plex Sans"', 'sans-serif'],
        mono: ['"IBM Plex Mono"', 'monospace'],
      },
      boxShadow: {
        panel: '0 1px 2px 0 rgba(15,23,42,0.04), 0 1px 3px 0 rgba(15,23,42,0.06)',
      },
    },
  },
  plugins: [],
}
