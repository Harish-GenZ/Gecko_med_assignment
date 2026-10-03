import React, { useRef, useState } from 'react';
import { UploadCloud, X, Image as ImageIcon, AlertCircle } from 'lucide-react';
import type { ImageMetadata } from '../types/verification';

interface ImageUploaderProps {
  imageMetadata: ImageMetadata | null;
  onImageSelected: (metadata: ImageMetadata | null) => void;
  disabled?: boolean;
}

const ACCEPTED_TYPES = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
const MAX_FILE_SIZE_MB = 15;

export const ImageUploader: React.FC<ImageUploaderProps> = ({
  imageMetadata,
  onImageSelected,
  disabled = false,
}) => {
  const [dragActive, setDragActive] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const processFile = (file: File) => {
    setUploadError(null);

    // Validate mime type
    if (!ACCEPTED_TYPES.includes(file.type.toLowerCase())) {
      setUploadError('Unsupported format. Please upload a JPEG, PNG, or WebP image.');
      return;
    }

    // Validate size
    const sizeMb = file.size / (1024 * 1024);
    if (sizeMb > MAX_FILE_SIZE_MB) {
      setUploadError(`File too large (${sizeMb.toFixed(1)} MB). Maximum allowed size is ${MAX_FILE_SIZE_MB} MB.`);
      return;
    }

    const previewUrl = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      onImageSelected({
        file,
        previewUrl,
        width: img.naturalWidth,
        height: img.naturalHeight,
        sizeKb: Math.round(file.size / 1024),
      });
    };
    img.onerror = () => {
      onImageSelected({
        file,
        previewUrl,
        sizeKb: Math.round(file.size / 1024),
      });
    };
    img.src = previewUrl;
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (disabled) return;
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (disabled) return;
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (disabled) return;
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const handleRemove = () => {
    if (imageMetadata?.previewUrl) {
      URL.revokeObjectURL(imageMetadata.previewUrl);
    }
    onImageSelected(null);
    setUploadError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <label className="text-xs font-semibold uppercase tracking-wider text-slate-400">
          Storefront Photograph <span className="text-rose-400">*</span>
        </label>
        {imageMetadata && (
          <span className="text-xs text-slate-400 font-mono">
            {imageMetadata.sizeKb} KB {imageMetadata.width && imageMetadata.height ? `(${imageMetadata.width}×${imageMetadata.height}px)` : ''}
          </span>
        )}
      </div>

      {!imageMetadata ? (
        <div
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => !disabled && fileInputRef.current?.click()}
          className={`relative border-2 border-dashed rounded-lg p-6 flex flex-col items-center justify-center text-center cursor-pointer transition-colors duration-150 ${
            dragActive
              ? 'border-indigo-500 bg-indigo-950/20'
              : 'border-slate-800 bg-slate-900/60 hover:border-slate-700 hover:bg-slate-900'
          } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp"
            onChange={handleChange}
            disabled={disabled}
            className="hidden"
            aria-label="Upload storefront photograph"
          />

          <div className="w-12 h-12 rounded-full bg-slate-800/80 flex items-center justify-center text-indigo-400 mb-3">
            <UploadCloud className="w-6 h-6" />
          </div>

          <p className="text-sm font-medium text-slate-200">
            Click to upload or drag & drop storefront photograph
          </p>
          <p className="text-xs text-slate-500 mt-1">
            JPEG, PNG, or WebP up to 15MB
          </p>
        </div>
      ) : (
        <div className="relative border border-slate-800 rounded-lg overflow-hidden bg-slate-900/80 p-3">
          <div className="flex items-start gap-3">
            <div className="relative w-24 h-24 rounded border border-slate-700 overflow-hidden bg-slate-950 flex-shrink-0 flex items-center justify-center">
              <img
                src={imageMetadata.previewUrl}
                alt="Storefront preview"
                className="w-full h-full object-cover"
              />
            </div>

            <div className="flex-1 min-w-0 flex flex-col justify-between py-0.5">
              <div>
                <p className="text-sm font-semibold text-slate-200 truncate" title={imageMetadata.file.name}>
                  {imageMetadata.file.name}
                </p>
                <div className="flex items-center gap-2 mt-1 text-xs text-slate-400">
                  <span className="inline-flex items-center gap-1">
                    <ImageIcon className="w-3.5 h-3.5 text-slate-500" />
                    {imageMetadata.file.type.split('/')[1]?.toUpperCase() || 'IMAGE'}
                  </span>
                  <span>•</span>
                  <span>{imageMetadata.sizeKb} KB</span>
                  {imageMetadata.width && imageMetadata.height && (
                    <>
                      <span>•</span>
                      <span>{imageMetadata.width} × {imageMetadata.height}</span>
                    </>
                  )}
                </div>
              </div>

              <div className="flex items-center gap-2 mt-3">
                <button
                  type="button"
                  onClick={() => !disabled && fileInputRef.current?.click()}
                  disabled={disabled}
                  className="text-xs text-indigo-400 hover:text-indigo-300 font-medium px-2 py-1 rounded bg-indigo-950/40 border border-indigo-900/60 hover:bg-indigo-900/40 transition-colors"
                >
                  Replace photo
                </button>
                <button
                  type="button"
                  onClick={handleRemove}
                  disabled={disabled}
                  className="text-xs text-rose-400 hover:text-rose-300 font-medium px-2 py-1 rounded bg-rose-950/40 border border-rose-900/60 hover:bg-rose-900/40 transition-colors inline-flex items-center gap-1"
                >
                  <X className="w-3.5 h-3.5" />
                  Remove
                </button>
              </div>
            </div>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp"
            onChange={handleChange}
            disabled={disabled}
            className="hidden"
          />
        </div>
      )}

      {uploadError && (
        <div className="flex items-center gap-1.5 text-xs text-rose-400 mt-1.5 bg-rose-950/40 border border-rose-900/50 p-2 rounded">
          <AlertCircle className="w-3.5 h-3.5 flex-shrink-0" />
          <span>{uploadError}</span>
        </div>
      )}
    </div>
  );
};
