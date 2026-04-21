# React + TypeScript + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Oxc](https://oxc.rs)
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/)

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the ESLint configuration

If you are developing a production application, we recommend updating the configuration to enable type-aware lint rules:

```js
export default defineConfig([
  # Rainfall Forecast Frontend (React)

  Frontend app for your FastAPI weather and rainfall prediction backend.

  ## Implemented Features

  - India map picker (click or drag marker) with latitude/longitude selection.
  - Current weather view using coordinates.
  - Weather forecast view (1 to 10 days).
  - Zonewise rainfall forecast with sensitivity slider (1 to 10).
  - Station-based rainfall forecast.
  - Result tables and charts for forecast outputs.

  ## API Endpoints Used

  - POST `/weather/current/`
  - POST `/forecast/weather/`
  - POST `/forecast/rainfall/zonewise/`
  - POST `/forecast/rainfall/`

  ## Setup

  1. Copy `.env.example` to `.env`.
  2. Set backend URL:

  `VITE_API_BASE_URL=http://localhost:8000`

  3. Install and run:

  - `npm install`
  - `npm run dev`

  ## Build

  - `npm run build`
  - `npm run preview`
    files: ['**/*.{ts,tsx}'],
