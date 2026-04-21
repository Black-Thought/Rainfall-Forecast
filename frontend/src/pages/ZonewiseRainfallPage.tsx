import { useEffect, useMemo, useState } from "react";
import dayjs from "dayjs";
import { useForm, useWatch } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { IndiaMapPicker } from "../components/IndiaMapPicker";
import { PageHeader } from "../components/PageHeader";
import { getStationRainfallForecast, getZonewiseRainfallForecast } from "../api/forecast";
import { getApiErrorMessage } from "../api/client";
import type {
  Coordinates,
  StationRainfallForecastResponse,
  ZonewiseRainfallForecastResponse,
} from "../api/types";
import { RainfallForecastChart } from "../components/charts/RainfallForecastChart";
import {
  filterStationNames,
  findStationByName,
  loadStations,
  type Station,
} from "../data/stations";

const defaultCoords: Coordinates = { lat: 22.9734, lon: 78.6569 };

const schema = z.object({
  lat: z.number().min(-90).max(90),
  lon: z.number().min(-180).max(180),
  start_date: z.string().min(1),
  num_days: z.number().min(1).max(30),
  sensitivity: z.number().min(1).max(10),
});

type FormValues = z.infer<typeof schema>;

export function ZonewiseRainfallPage() {
  const [coords, setCoords] = useState<Coordinates>(defaultCoords);
  const [result, setResult] = useState<ZonewiseRainfallForecastResponse | null>(null);
  const [stationResult, setStationResult] = useState<StationRainfallForecastResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stations, setStations] = useState<Station[]>([]);
  const [stationQuery, setStationQuery] = useState("");
  const [stationDays, setStationDays] = useState(7);
  const [stationDate, setStationDate] = useState(dayjs().format("YYYY-MM-DD"));
  const [stationError, setStationError] = useState<string | null>(null);
  const [stationLoading, setStationLoading] = useState(false);

  const {
    register,
    handleSubmit,
    setValue,
    control,
    formState: { errors },
  } = useForm<FormValues>({
    resolver: zodResolver(schema),
    defaultValues: {
      lat: defaultCoords.lat,
      lon: defaultCoords.lon,
      start_date: dayjs().format("YYYY-MM-DD"),
      num_days: 7,
      sensitivity: 5,
    },
  });

  const sensitivityValue = useWatch({
    control,
    name: "sensitivity",
  });

  useEffect(() => {
    loadStations()
      .then((items) => setStations(items))
      .catch(() => setStations([]));
  }, []);

  const stationSuggestions = useMemo(
    () => filterStationNames(stations, stationQuery),
    [stations, stationQuery],
  );

  const onSubmit = async (values: FormValues) => {
    setLoading(true);
    setError(null);
    try {
      const data = await getZonewiseRainfallForecast({
        location: { lat: values.lat, lon: values.lon },
        start_date: values.start_date,
        num_days: values.num_days,
        sensitivity: values.sensitivity,
      });
      setResult(data);
      setCoords({ lat: values.lat, lon: values.lon });
    } catch (err) {
      setError(getApiErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  const runStationForecast = async () => {
    setStationLoading(true);
    setStationError(null);
    try {
      const data = await getStationRainfallForecast({
        station_name: stationQuery,
        start_date: stationDate,
        num_days: stationDays,
      });
      setStationResult(data);

      const selectedStation = findStationByName(stations, stationQuery);
      if (selectedStation) {
        const next = { lat: selectedStation.latitude, lon: selectedStation.longitude };
        setCoords(next);
        setValue("lat", Number(next.lat.toFixed(4)));
        setValue("lon", Number(next.lon.toFixed(4)));
      }
    } catch (err) {
      setStationError(getApiErrorMessage(err));
    } finally {
      setStationLoading(false);
    }
  };

  return (
    <div>
      <PageHeader
        title="Zonewise Rainfall Forecast"
        description="Pick location, date range and sensitivity (1-10 nearest stations)."
      />

      <section className="panel">
        <form onSubmit={handleSubmit(onSubmit)} className="form-grid">
          <label>
            Latitude
            <input type="number" step="0.0001" {...register("lat", { valueAsNumber: true })} />
            {errors.lat ? <small className="error">{errors.lat.message}</small> : null}
          </label>

          <label>
            Longitude
            <input type="number" step="0.0001" {...register("lon", { valueAsNumber: true })} />
            {errors.lon ? <small className="error">{errors.lon.message}</small> : null}
          </label>

          <label>
            Start Date
            <input type="date" {...register("start_date")} />
            {errors.start_date ? <small className="error">{errors.start_date.message}</small> : null}
          </label>

          <label>
            Forecast Days
            <input type="number" {...register("num_days", { valueAsNumber: true })} />
            {errors.num_days ? <small className="error">{errors.num_days.message}</small> : null}
          </label>

          <label>
            Sensitivity: {sensitivityValue}
            <input
              type="range"
              min={1}
              max={10}
              step={1}
              {...register("sensitivity", { valueAsNumber: true })}
            />
            {errors.sensitivity ? <small className="error">{errors.sensitivity.message}</small> : null}
          </label>

          <button type="submit" disabled={loading}>
            {loading ? "Forecasting..." : "Get Zonewise Forecast"}
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
          <div className="result-grid">
            <div className="stat">
              <h4>Dominant Zone</h4>
              <p>{result.dominant_zone}</p>
            </div>
            <div className="stat">
              <h4>Sensitivity</h4>
              <p>{result.sensitivity}</p>
            </div>
            <div className="stat">
              <h4>Days</h4>
              <p>{result.num_days}</p>
            </div>
          </div>

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

      <section className="panel">
        <h3>Station-wise Rainfall Search</h3>
        <p className="hint">Type station name for suggestions, or click a station marker on map.</p>
        <div className="form-grid">
          <label>
            Station Name
            <input
              list="stations-list"
              value={stationQuery}
              onChange={(e) => setStationQuery(e.target.value)}
              placeholder="Start typing station name"
            />
            <datalist id="stations-list">
              {stationSuggestions.map((name) => (
                <option key={name} value={name} />
              ))}
            </datalist>
          </label>

          <label>
            Start Date
            <input type="date" value={stationDate} onChange={(e) => setStationDate(e.target.value)} />
          </label>

          <label>
            Forecast Days
            <input
              type="number"
              min={1}
              max={30}
              value={stationDays}
              onChange={(e) => setStationDays(Number(e.target.value))}
            />
          </label>

          <button type="button" onClick={runStationForecast} disabled={stationLoading || !stationQuery}>
            {stationLoading ? "Forecasting..." : "Get Station Forecast"}
          </button>
        </div>

        {stationError ? <p className="error">{stationError}</p> : null}

        {stationResult ? (
          <>
            <div className="table-wrapper">
              <table>
                <thead>
                  <tr>
                    <th>Date</th>
                    <th>Forecasted Rainfall (mm)</th>
                  </tr>
                </thead>
                <tbody>
                  {stationResult.predictions.map((p) => (
                    <tr key={p.date_of_record}>
                      <td>{p.date_of_record}</td>
                      <td>{p.predicted_rainfall.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <RainfallForecastChart data={stationResult.predictions} />
          </>
        ) : null}
      </section>
    </div>
  );
}
