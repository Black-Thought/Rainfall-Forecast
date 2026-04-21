import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { WeatherForecastDay } from "../../api/types";

type Props = {
  data: WeatherForecastDay[];
};

export function WeatherForecastChart({ data }: Props) {
  return (
    <div className="chart-card">
      <h3>Temperature and Precipitation Trend</h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="avg_temp_c" name="Avg Temp (°C)" stroke="#1f77b4" />
          <Line
            type="monotone"
            dataKey="total_precip_mm"
            name="Precip (mm)"
            stroke="#2ca02c"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
