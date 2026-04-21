export type Coordinates = {
  lat: number;
  lon: number;
};

export type WeatherForecastDay = {
  date: string;
  avg_temp_c: number;
  max_temp_c: number;
  min_temp_c: number;
  total_precip_mm: number;
  rain_probability: number;
  will_rain: boolean;
  max_wind_kph: number;
  condition: string;
};

export type WeatherForecastResponse = {
  location: Coordinates;
  resolved_name: string;
  forecast: WeatherForecastDay[];
};

export type CurrentWeatherResponse = {
  location: string;
  region: string;
  country: string;
  lat: number;
  lon: number;
  temperature_c: number;
  feels_like_c: number;
  humidity: number;
  pressure_mb: number;
  visibility_km: number;
  wind_kph: number;
  wind_degree: number;
  wind_direction: string;
  condition: string;
  cloud: number;
  uv: number;
  precip_mm: number;
  is_raining: boolean;
  aqi: {
    pm2_5: number | null;
    pm10: number | null;
    co: number | null;
    no2: number | null;
    o3: number | null;
    so2: number | null;
  };
};

export type ZonewiseRainfallForecastItem = {
  date_of_record: string;
  predicted_rainfall: number;
};

export type ZonewiseRainfallForecastResponse = {
  location: Coordinates;
  dominant_zone: string;
  start_date: string;
  num_days: number;
  sensitivity: number;
  predictions: ZonewiseRainfallForecastItem[];
};

export type StationRainfallForecastItem = {
  date_of_record: string;
  predicted_rainfall: number;
};

export type StationRainfallForecastResponse = {
  station_name: string;
  start_date: string;
  num_days: number;
  predictions: StationRainfallForecastItem[];
};
