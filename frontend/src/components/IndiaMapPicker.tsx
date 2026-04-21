import L from "leaflet";
import { MapContainer, Marker, TileLayer, useMapEvents } from "react-leaflet";
import { useMemo } from "react";
import type { Coordinates } from "../api/types";

type Props = {
  value: Coordinates;
  onChange: (coords: Coordinates) => void;
};

const INDIA_BOUNDS: [[number, number], [number, number]] = [
  [6, 68],
  [38, 98],
];

function ClickHandler({ onChange }: { onChange: (coords: Coordinates) => void }) {
  useMapEvents({
    click(e) {
      onChange({ lat: e.latlng.lat, lon: e.latlng.lng });
    },
  });

  return null;
}

const selectedIcon = L.divIcon({
  className: "selected-marker",
  html: "<span></span>",
  iconSize: [20, 20],
  iconAnchor: [10, 10],
});

export function IndiaMapPicker({ value, onChange }: Props) {
  const markerPosition = useMemo<[number, number]>(
    () => [value.lat, value.lon],
    [value.lat, value.lon],
  );

  return (
    <MapContainer
      center={markerPosition}
      zoom={5}
      minZoom={4}
      maxZoom={10}
      maxBounds={INDIA_BOUNDS}
      className="map"
      scrollWheelZoom
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      <ClickHandler onChange={onChange} />
      <Marker
        position={markerPosition}
        icon={selectedIcon}
        draggable
        eventHandlers={{
          dragend: (event) => {
            const marker = event.target;
            const { lat, lng } = marker.getLatLng();
            onChange({ lat, lon: lng });
          },
        }}
      />
    </MapContainer>
  );
}
