import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { IndiaMapPicker } from "../components/IndiaMapPicker";
import { PageHeader } from "../components/PageHeader";
import { getWeatherForecast } from "../api/forecast";
import { getApiErrorMessage } from "../api/client";
import type { Coordinates, WeatherForecastResponse } from "../api/types";
import { WeatherForecastChart } from "../components/charts/WeatherForecastChart";

const schema = z.object({
  lat: z.number().min(-90).max(90),
  lon: z.number().min(-180).max(180),
  num_days: z.number().min(1).max(10),
});

type FormValues = z.infer<typeof schema>;

const defaultCoords: Coordinates = { lat: 22.9734, lon: 78.6569 };

export function WeatherForecastPage() {
  const [coords, setCoords] = useState<Coordinates>(defaultCoords);
  const [result, setResult] = useState<WeatherForecastResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const {
    register,
    handleSubmit,
    setValue,
    formState: { errors },
  } = useForm<FormValues>({
    resolver: zodResolver(schema),
    defaultValues: {
      lat: defaultCoords.lat,
      lon: defaultCoords.lon,
      num_days: 5,
    },
  });

  const onSubmit = async (values: FormValues) => {
    setLoading(true);
    setError(null);
    try {
      const data = await getWeatherForecast({
        coordinates: {
          lat: values.lat,
          lon: values.lon,
        },
        num_days: values.num_days,
      });
      setResult(data);
      setCoords({ lat: values.lat, lon: values.lon });
    } catch (err) {
      setError(getApiErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <PageHeader
        title="Weather Forecast"
        description="Predict weather for up to 10 days using coordinates from India map."
      />

      <section className="panel">
        <form onSubmit={handleSubmit(onSubmit)} className="form-grid">
          <label>
            Latitude
            <input
              type="number"
              step="0.0001"
              {...register("lat", { valueAsNumber: true })}
            />
            {errors.lat ? <small className="error">{errors.lat.message}</small> : null}
          </label>

          <label>
            Longitude
            <input
              type="number"
              step="0.0001"
              {...register("lon", { valueAsNumber: true })}
            />
            {errors.lon ? <small className="error">{errors.lon.message}</small> : null}
          </label>

          <label>
            Forecast days (1-10)
            <input type="number" {...register("num_days", { valueAsNumber: true })} />
            {errors.num_days ? <small className="error">{errors.num_days.message}</small> : null}
          </label>

          <button type="submit" disabled={loading}>
            {loading ? "Forecasting..." : "Get Forecast"}
          </button>
        </form>

        <IndiaMapPicker
          value={coords}
          onChange={(next) => {
            setCoords(next);
            setValue("lat", Number(next.lat.toFixed(4)));
            setValue("lon", Number(next.lon.toFixed(4)));
          }}
        />
      </section>

      {error ? <p className="error">{error}</p> : null}

      {result ? (
        <section className="panel">
          <h3>Resolved location: {result.resolved_name}</h3>
          <div className="table-wrapper">
            <table>
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Avg Temp (°C)</th>
                  <th>Rain Probability (%)</th>
                  <th>Precip (mm)</th>
                  <th>Condition</th>
                </tr>
              </thead>
              <tbody>
                {result.forecast.map((day) => (
                  <tr key={day.date}>
                    <td>{day.date}</td>
                    <td>{day.avg_temp_c}</td>
                    <td>{day.rain_probability}</td>
                    <td>{day.total_precip_mm}</td>
                    <td>{day.condition}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <WeatherForecastChart data={result.forecast} />
        </section>
      ) : null}
    </div>
  );
}
