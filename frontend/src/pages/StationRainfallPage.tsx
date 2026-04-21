import { useState } from "react";
import dayjs from "dayjs";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { getStationRainfallForecast } from "../api/forecast";
import { getApiErrorMessage } from "../api/client";
import type { StationRainfallForecastResponse } from "../api/types";
import { PageHeader } from "../components/PageHeader";
import { RainfallForecastChart } from "../components/charts/RainfallForecastChart";

const schema = z.object({
  station_name: z.string().min(1, "Station name is required"),
  start_date: z.string().min(1),
  num_days: z.number().min(1).max(30),
});

type FormValues = z.infer<typeof schema>;

export function StationRainfallPage() {
  const [result, setResult] = useState<StationRainfallForecastResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<FormValues>({
    resolver: zodResolver(schema),
    defaultValues: {
      station_name: "",
      start_date: dayjs().format("YYYY-MM-DD"),
      num_days: 7,
    },
  });

  const onSubmit = async (values: FormValues) => {
    setLoading(true);
    setError(null);
    try {
      const data = await getStationRainfallForecast(values);
      setResult(data);
    } catch (err) {
      setError(getApiErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <PageHeader
        title="Station Rainfall Forecast"
        description="Predict rainfall by station name and forecast period."
      />

      <section className="panel">
        <form onSubmit={handleSubmit(onSubmit)} className="form-grid">
          <label>
            Station Name
            <input placeholder="e.g. Pune" {...register("station_name")} />
            {errors.station_name ? <small className="error">{errors.station_name.message}</small> : null}
          </label>

          <label>
            Start Date
            <input type="date" {...register("start_date")} />
          </label>

          <label>
            Forecast Days
            <input type="number" {...register("num_days", { valueAsNumber: true })} />
          </label>

          <button type="submit" disabled={loading}>
            {loading ? "Forecasting..." : "Get Station Forecast"}
          </button>
        </form>
      </section>

      {error ? <p className="error">{error}</p> : null}

      {result ? (
        <section className="panel">
          <h3>Station: {result.station_name}</h3>
          <div className="table-wrapper">
            <table>
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Forecasted Rainfall (mm)</th>
                </tr>
              </thead>
              <tbody>
                {result.predictions.map((p) => (
                  <tr key={p.date_of_record}>
                    <td>{p.date_of_record}</td>
                    <td>{p.predicted_rainfall.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <RainfallForecastChart data={result.predictions} />
        </section>
      ) : null}
    </div>
  );
}
