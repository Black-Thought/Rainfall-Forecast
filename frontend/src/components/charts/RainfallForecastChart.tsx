import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type RainPoint = {
  date_of_record: string;
  predicted_rainfall: number;
};

type Props = {
  data: RainPoint[];
};

export function RainfallForecastChart({ data }: Props) {
  return (
    <div className="chart-card">
      <h3>Forecasted Rainfall (mm)</h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date_of_record" />
          <YAxis />
          <Tooltip />
          <Line
            type="monotone"
            dataKey="predicted_rainfall"
            stroke="#0ea5e9"
            strokeWidth={2}
            dot={{ r: 2 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
