import { Link } from "react-router-dom";

const cards = [
  {
    title: "Current Weather",
    description: "Select coordinates on map and get current weather + AQI.",
    href: "/current-weather",
  },
  {
    title: "Weather Forecast",
    description: "Forecast weather for next 1-10 days using coordinates.",
    href: "/weather-forecast",
  },
  {
    title: "Zonewise Rainfall",
    description: "Rainfall forecast by monsoon zone with sensitivity control.",
    href: "/zonewise-rainfall",
  },
  {
    title: "Station Rainfall",
    description: "Forecast rainfall by station name and date range.",
    href: "/station-rainfall",
  },
];

export function DashboardPage() {
  return (
    <div>
      <div className="hero">
        <h1>India Weather Intelligence</h1>
        <p>
          Pick any location on the India map and run weather or rainfall prediction from your
          backend APIs.
        </p>
      </div>
      <div className="grid cards-grid">
        {cards.map((card) => (
          <Link to={card.href} className="card" key={card.href}>
            <h3>{card.title}</h3>
            <p>{card.description}</p>
            <span>Open</span>
          </Link>
        ))}
      </div>
    </div>
  );
}
