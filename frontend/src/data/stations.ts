export type Station = {
  station_name: string;
  state: string;
  district: string;
  latitude: number;
  longitude: number;
};

function parseStationCsv(csvText: string): Station[] {
  const lines = csvText
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  if (lines.length <= 1) return [];

  return lines.slice(1).map((line) => {
    const [station_name, state, district, latitude, longitude] = line.split(",");
    return {
      station_name,
      state,
      district,
      latitude: Number(latitude),
      longitude: Number(longitude),
    };
  });
}

let stationCache: Station[] | null = null;

export async function loadStations(): Promise<Station[]> {
  if (stationCache) return stationCache;

  const response = await fetch("/stations_coordinates.csv");
  if (!response.ok) {
    throw new Error("Failed to load stations data");
  }

  const csvText = await response.text();
  stationCache = parseStationCsv(csvText).filter(
    (item) => Number.isFinite(item.latitude) && Number.isFinite(item.longitude),
  );

  return stationCache;
}

export function filterStationNames(stations: Station[], query: string): string[] {
  const q = query.trim().toLowerCase();
  if (!q) {
    return stations.slice(0, 20).map((s) => s.station_name);
  }

  return stations
    .filter((s) => s.station_name.toLowerCase().includes(q))
    .slice(0, 20)
    .map((s) => s.station_name);
}

export function findStationByName(stations: Station[], name: string): Station | undefined {
  const cleaned = name.trim().toLowerCase();
  return stations.find((s) => s.station_name.trim().toLowerCase() === cleaned);
}
