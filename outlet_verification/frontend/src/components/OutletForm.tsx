import React, { useState } from 'react';
import { MapPin, Navigation, ShieldCheck } from 'lucide-react';
import { ImageUploader } from './ImageUploader';
import type { ImageMetadata } from '../types/verification';

interface OutletFormProps {
  onSubmit: (data: { name: string; latitude: number; longitude: number; imageFile: File; autoRegister: boolean }) => void;
  isLoading: boolean;
}



export const OutletForm: React.FC<OutletFormProps> = ({ onSubmit, isLoading }) => {
  const [name, setName] = useState('');
  const [latitude, setLatitude] = useState('');
  const [longitude, setLongitude] = useState('');
  const [imageMetadata, setImageMetadata] = useState<ImageMetadata | null>(null);
  const [autoRegister, setAutoRegister] = useState(true);

  const [geoError, setGeoError] = useState<string | null>(null);
  const [isLocating, setIsLocating] = useState(false);

  // Form validation
  const parsedLat = parseFloat(latitude);
  const parsedLng = parseFloat(longitude);
  const isLatValid = !isNaN(parsedLat) && parsedLat >= -90 && parsedLat <= 90;
  const isLngValid = !isNaN(parsedLng) && parsedLng >= -180 && parsedLng <= 180;
  const isNameValid = name.trim().length > 0;
  const isImageValid = imageMetadata !== null;

  const isFormValid = isNameValid && isLatValid && isLngValid && isImageValid;

  const handleGetCurrentLocation = () => {
    setGeoError(null);
    if (!navigator.geolocation) {
      setGeoError('Geolocation is not supported by your browser.');
      return;
    }

    setIsLocating(true);
    navigator.geolocation.getCurrentPosition(
      (position) => {
        setLatitude(position.coords.latitude.toFixed(6));
        setLongitude(position.coords.longitude.toFixed(6));
        setIsLocating(false);
      },
      (error) => {
        setIsLocating(false);
        switch (error.code) {
          case error.PERMISSION_DENIED:
            setGeoError('Location permission denied. Please enter coordinates manually.');
            break;
          case error.POSITION_UNAVAILABLE:
            setGeoError('Location information is unavailable.');
            break;
          case error.TIMEOUT:
            setGeoError('The request to get user location timed out.');
            break;
          default:
            setGeoError('An unknown error occurred while retrieving location.');
            break;
        }
      },
      { timeout: 10000, enableHighAccuracy: true }
    );
  };


  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!isFormValid || !imageMetadata) return;

    onSubmit({
      name: name.trim(),
      latitude: parsedLat,
      longitude: parsedLng,
      imageFile: imageMetadata.file,
      autoRegister,
    });
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">


      {/* Outlet Name */}
      <div className="space-y-1.5">
        <label htmlFor="outlet-name" className="block text-xs font-semibold uppercase tracking-wider text-slate-400">
          Outlet Name <span className="text-rose-400">*</span>
        </label>
        <input
          id="outlet-name"
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="e.g., Nilgiris Supermarket, Sri Murugan General Stores, or மளிகை கடை"
          disabled={isLoading}
          required
          className="w-full px-3.5 py-2.5 bg-slate-900 border border-slate-800 rounded-lg text-slate-100 placeholder-slate-500 text-sm focus:outline-none focus:ring-1 focus:ring-indigo-500 focus:border-indigo-500 transition-colors"
        />
      </div>

      {/* Storefront Image */}
      <ImageUploader
        imageMetadata={imageMetadata}
        onImageSelected={setImageMetadata}
        disabled={isLoading}
      />

      {/* GPS Coordinates */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <label className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Geographic Coordinates <span className="text-rose-400">*</span>
          </label>
          <button
            type="button"
            onClick={handleGetCurrentLocation}
            disabled={isLoading || isLocating}
            className="inline-flex items-center gap-1 text-xs text-indigo-400 hover:text-indigo-300 font-medium px-2 py-0.5 rounded hover:bg-indigo-950/40 transition-colors"
          >
            <Navigation className={`w-3 h-3 ${isLocating ? 'animate-spin' : ''}`} />
            <span>{isLocating ? 'Locating...' : 'Use Current Location'}</span>
          </button>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <div className="relative">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-slate-500">
              <MapPin className="w-3.5 h-3.5" />
            </div>
            <input
              type="number"
              step="any"
              value={latitude}
              onChange={(e) => {
                setLatitude(e.target.value);
                setGeoError(null);
              }}
              placeholder="Latitude (e.g., 12.9716)"
              disabled={isLoading}
              required
              aria-label="Latitude coordinate"
              className="w-full pl-9 pr-3 py-2 bg-slate-900 border border-slate-800 rounded-lg text-slate-100 placeholder-slate-500 text-sm focus:outline-none focus:ring-1 focus:ring-indigo-500 focus:border-indigo-500 font-mono transition-colors"
            />
          </div>

          <div className="relative">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-slate-500">
              <MapPin className="w-3.5 h-3.5" />
            </div>
            <input
              type="number"
              step="any"
              value={longitude}
              onChange={(e) => {
                setLongitude(e.target.value);
                setGeoError(null);
              }}
              placeholder="Longitude (e.g., 77.5946)"
              disabled={isLoading}
              required
              aria-label="Longitude coordinate"
              className="w-full pl-9 pr-3 py-2 bg-slate-900 border border-slate-800 rounded-lg text-slate-100 placeholder-slate-500 text-sm focus:outline-none focus:ring-1 focus:ring-indigo-500 focus:border-indigo-500 font-mono transition-colors"
            />
          </div>
        </div>

        {geoError && (
          <p className="text-xs text-rose-400 mt-1">{geoError}</p>
        )}
      </div>

      {/* Auto-Registration Toggle */}
      <div className="flex items-center justify-between p-3 bg-slate-900/60 border border-slate-800 rounded-lg">
        <div className="flex flex-col">
          <span className="text-xs font-semibold text-slate-200">Auto-Register Genuine Outlets</span>
          <span className="text-[11px] text-slate-400">If verified as genuine retail outlet, upload image to Railway Bucket & save to PostgreSQL</span>
        </div>
        <label className="relative inline-flex items-center cursor-pointer">
          <input
            type="checkbox"
            checked={autoRegister}
            onChange={(e) => setAutoRegister(e.target.checked)}
            disabled={isLoading}
            className="sr-only peer"
          />
          <div className="w-9 h-5 bg-slate-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-emerald-600"></div>
        </label>
      </div>

      {/* Submit Button */}
      <div className="pt-2">
        <button
          type="submit"
          disabled={!isFormValid || isLoading}
          className={`w-full py-3 px-4 rounded-lg font-semibold text-sm flex items-center justify-center gap-2 transition-all duration-150 ${
            isFormValid && !isLoading
              ? 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-lg shadow-indigo-600/20 active:scale-[0.99] cursor-pointer'
              : 'bg-slate-800 text-slate-500 border border-slate-700/50 cursor-not-allowed'
          }`}
        >
          <ShieldCheck className="w-4 h-4" />
          <span>{isLoading ? 'Processing Verification...' : 'Verify Outlet with AI Engine'}</span>
        </button>

        {!isFormValid && (
          <p className="text-center text-xs text-slate-500 mt-2">
            {!isNameValid
              ? 'Enter outlet name'
              : !isImageValid
              ? 'Upload storefront photo'
              : 'Enter valid latitude and longitude'}{' '}
            to proceed.
          </p>
        )}
      </div>
    </form>
  );
};
