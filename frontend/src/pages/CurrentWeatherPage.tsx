import { useState } from "react";
import { IndiaMapPicker } from "../components/IndiaMapPicker";
import { PageHeader } from "../components/PageHeader";
import { getCurrentWeather } from "../api/forecast";
import type { Coordinates, CurrentWeatherResponse } from "../api/types";
import { getApiErrorMessage } from "../api/client";

const defaultCoords: Coordinates = { lat: 22.9734, lon: 78.6569 };

export function CurrentWeatherPage() {
  const [coords, setCoords] = useState<Coordinates>(defaultCoords);
  const [result, setResult] = useState<CurrentWeatherResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit() {
    setLoading(true);
    setError(null);
    try {
      const data = await getCurrentWeather(coords);
      setResult(data);
    } catch (err) {
      setError(getApiErrorMessage(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <PageHeader
        title="Current Weather"
        description="Pick coordinates from map to fetch real-time weather and AQI."
      />

      <section className="panel">
        <div className="coords-row">
          <label>
            Latitude
            <input
              type="number"
              step="0.0001"
              value={coords.lat}
              onChange={(e) => setCoords({ ...coords, lat: Number(e.target.value) })}
            />
          </label>
          <label>
            Longitude
            <input
              type="number"
              step="0.0001"
              value={coords.lon}
              onChange={(e) => setCoords({ ...coords, lon: Number(e.target.value) })}
            />
          </label>
          <button onClick={submit} disabled={loading}>
            {loading ? "Loading..." : "Get Current Weather"}
          </button>
        </div>

        <IndiaMapPicker value={coords} onChange={setCoords} />
      </section>

      {error ? <p className="error">{error}</p> : null}

      {result ? (
        <section className="panel result-grid">
          <div className="stat">
            <h3>{result.location}</h3>
            <p>
              {result.region}, {result.country}
            </p>
          </div>
          <div className="stat">
            <h4>Temperature</h4>
            <p>{result.temperature_c} °C</p>
          </div>
          <div className="stat">
            <h4>Feels Like</h4>
            <p>{result.feels_like_c} °C</p>
          </div>
          <div className="stat">
            <h4>Humidity</h4>
            <p>{result.humidity}%</p>
          </div>
          <div className="stat">
            <h4>Condition</h4>
            <p>{result.condition}</p>
          </div>
          <div className="stat">
            <h4>AQI PM2.5</h4>
            <p>{result.aqi.pm2_5 ?? "N/A"}</p>
          </div>
        </section>
      ) : null}
    </div>
  );
}
