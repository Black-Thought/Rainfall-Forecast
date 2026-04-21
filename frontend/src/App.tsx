import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import { CurrentWeatherPage } from "./pages/CurrentWeatherPage";
import { DashboardPage } from "./pages/DashboardPage";
import { StationRainfallPage } from "./pages/StationRainfallPage";
import { WeatherForecastPage } from "./pages/WeatherForecastPage";
import { ZonewiseRainfallPage } from "./pages/ZonewiseRainfallPage";

function App() {
  return (
    <div className="layout">
      <nav className="top-nav">
        <div className="brand">Rainfall Forecast UI</div>
        <div className="links">
          <NavLink to="/" end>
            Dashboard
          </NavLink>
          <NavLink to="/current-weather">Current Weather</NavLink>
          <NavLink to="/weather-forecast">Weather Forecast</NavLink>
          <NavLink to="/zonewise-rainfall">Zonewise Rainfall</NavLink>
          <NavLink to="/station-rainfall">Station Rainfall</NavLink>
        </div>
      </nav>

      <main className="container">
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/current-weather" element={<CurrentWeatherPage />} />
          <Route path="/weather-forecast" element={<WeatherForecastPage />} />
          <Route path="/zonewise-rainfall" element={<ZonewiseRainfallPage />} />
          <Route path="/station-rainfall" element={<StationRainfallPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;
