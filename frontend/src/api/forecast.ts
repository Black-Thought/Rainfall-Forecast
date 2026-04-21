import { apiClient } from "./client";
import type {
  Coordinates,
  CurrentWeatherResponse,
  StationRainfallForecastResponse,
  WeatherForecastResponse,
  ZonewiseRainfallForecastResponse,
} from "./types";

export async function getCurrentWeather(
  coordinates: Coordinates,
): Promise<CurrentWeatherResponse> {
  const response = await apiClient.post<CurrentWeatherResponse>("/weather/current/", {
    coordinates,
  });
  return response.data;
}

export async function getWeatherForecast(payload: {
  coordinates: Coordinates;
  num_days: number;
}): Promise<WeatherForecastResponse> {
  const response = await apiClient.post<WeatherForecastResponse>(
    "/forecast/weather/",
    payload,
  );
  return response.data;
}

export async function getZonewiseRainfallForecast(payload: {
  location: Coordinates;
  start_date: string;
  num_days: number;
  sensitivity: number;
}): Promise<ZonewiseRainfallForecastResponse> {
  const response = await apiClient.post<ZonewiseRainfallForecastResponse>(
    "/forecast/rainfall/zonewise/",
    payload,
  );
  return response.data;
}

export async function getStationRainfallForecast(payload: {
  station_name: string;
  start_date: string;
  num_days: number;
}): Promise<StationRainfallForecastResponse> {
  const response = await apiClient.post<StationRainfallForecastResponse>(
    "/forecast/rainfall/",
    payload,
  );
  return response.data;
}
