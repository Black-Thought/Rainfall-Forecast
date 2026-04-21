import {
  Bar,
  BarChart,
  CartesianGrid,
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
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date_of_record" />
          <YAxis />
          <Tooltip />
          <Bar dataKey="predicted_rainfall" fill="#0ea5e9" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
