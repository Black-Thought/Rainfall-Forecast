import axios from "axios";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL?.trim() || "http://localhost:8000";

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 15000,
  headers: {
    "Content-Type": "application/json",
  },
});

export function getApiErrorMessage(error: unknown): string {
  if (!axios.isAxiosError(error)) {
    return "Unexpected error occurred.";
  }

  const detail = error.response?.data?.detail;
  if (typeof detail === "string") {
    return detail;
  }

  if (Array.isArray(detail) && detail.length > 0) {
    return detail[0]?.msg || "Request validation error.";
  }

  return error.message || "Request failed.";
}
