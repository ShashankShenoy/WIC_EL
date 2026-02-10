import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, FileText, X } from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { cn } from '@/lib/utils'

interface FileUploadProps {
  onFilesSelected: (csvFile: File, pklFile: File) => void
}

export function FileUpload({ onFilesSelected }: FileUploadProps) {
  const [csvFile, setCsvFile] = useState<File | null>(null)
  const [pklFile, setPklFile] = useState<File | null>(null)

  const onDropCsv = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      setCsvFile(acceptedFiles[0])
    }
  }, [])

  const onDropPkl = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      setPklFile(acceptedFiles[0])
    }
  }, [])

  const { getRootProps: getCsvRootProps, getInputProps: getCsvInputProps, isDragActive: isCsvDragActive } = useDropzone({
    onDrop: onDropCsv,
    accept: { 'text/csv': ['.csv'] },
    multiple: false
  })

  const { getRootProps: getPklRootProps, getInputProps: getPklInputProps, isDragActive: isPklDragActive } = useDropzone({
    onDrop: onDropPkl,
    accept: { 'application/octet-stream': ['.pkl'] },
    multiple: false
  })

  const handleSubmit = () => {
    if (csvFile && pklFile) {
      onFilesSelected(csvFile, pklFile)
    }
  }

  return (
    <div className="space-y-4">
      <div
        {...getCsvRootProps()}
        className={cn(
          "cursor-pointer rounded-lg border-2 border-dashed p-8 text-center transition-all duration-300 group",
          isCsvDragActive ? "border-zinc-500 bg-zinc-900 scale-105" : "border-zinc-700 hover:border-zinc-600 bg-zinc-900/50",
          csvFile && "border-zinc-500 bg-zinc-900"
        )}
      >
        <input {...getCsvInputProps()} />
        <div className="inline-block p-3 bg-zinc-800 rounded-xl group-hover:bg-zinc-700 transition-colors border border-zinc-700">
          <Upload className="h-8 w-8 text-zinc-400 group-hover:text-zinc-300 transition-colors" />
        </div>
        <p className="mt-3 text-sm font-medium text-white">
          {csvFile ? csvFile.name : "Drop CSV file here or click to browse"}
        </p>
        <p className="text-xs text-zinc-500 mt-1">Historical carbon intensity data</p>
        {csvFile && (
          <Button
            variant="ghost"
            size="sm"
            className="mt-3"
            onClick={(e) => {
              e.stopPropagation()
              setCsvFile(null)
            }}
          >
            <X className="h-4 w-4 mr-1" /> Remove
          </Button>
        )}
      </div>

      <div
        {...getPklRootProps()}
        className={cn(
          "cursor-pointer rounded-lg border-2 border-dashed p-8 text-center transition-all duration-300 group",
          isPklDragActive ? "border-zinc-500 bg-zinc-900 scale-105" : "border-zinc-700 hover:border-zinc-600 bg-zinc-900/50",
          pklFile && "border-zinc-500 bg-zinc-900"
        )}
      >
        <input {...getPklInputProps()} />
        <div className="inline-block p-3 bg-zinc-800 rounded-xl group-hover:bg-zinc-700 transition-colors border border-zinc-700">
          <FileText className="h-8 w-8 text-zinc-400 group-hover:text-zinc-300 transition-colors" />
        </div>
        <p className="mt-3 text-sm font-medium text-white">
          {pklFile ? pklFile.name : "Drop PKL file here or click to browse"}
        </p>
        <p className="text-xs text-zinc-500 mt-1">Trained ML models (.pkl)</p>
        {pklFile && (
          <Button
            variant="ghost"
            size="sm"
            className="mt-3"
            onClick={(e) => {
              e.stopPropagation()
              setPklFile(null)
            }}
          >
            <X className="h-4 w-4 mr-1" /> Remove
          </Button>
        )}
      </div>

      <Button 
        className="w-full" 
        onClick={handleSubmit}
        disabled={!csvFile || !pklFile}
      >
        Upload Files
      </Button>
    </div>
  )
}
