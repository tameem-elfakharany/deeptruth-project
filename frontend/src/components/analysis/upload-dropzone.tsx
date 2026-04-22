"use client"

import { useCallback, useRef, useState } from "react"
import { ImagePlus, Mic, Video } from "lucide-react"

import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"

export function UploadDropzone({
  onFileSelected,
  accept = "image/*",
  note,
  title = "Drag and drop an image"
}: {
  onFileSelected: (file: File) => void
  accept?: string
  note?: string
  title?: string
}) {
  const inputRef = useRef<HTMLInputElement | null>(null)
  const [isDragging, setIsDragging] = useState(false)

  const openPicker = useCallback(() => {
    inputRef.current?.click()
  }, [])

  return (
    <div
      className={cn(
        "rounded-2xl border border-dashed bg-white p-8 text-center shadow-soft transition-colors",
        isDragging ? "border-brand-500 bg-brand-50" : "border-slate-200"
      )}
      onDragEnter={(e) => {
        e.preventDefault()
        e.stopPropagation()
        setIsDragging(true)
      }}
      onDragOver={(e) => {
        e.preventDefault()
        e.stopPropagation()
        setIsDragging(true)
      }}
      onDragLeave={(e) => {
        e.preventDefault()
        e.stopPropagation()
        setIsDragging(false)
      }}
      onDrop={(e) => {
        e.preventDefault()
        e.stopPropagation()
        setIsDragging(false)
        const file = e.dataTransfer.files?.[0]
        if (file) onFileSelected(file)
      }}
    >
      <div className="mx-auto grid h-14 w-14 place-items-center rounded-2xl bg-slate-100 text-slate-700">
          {accept.includes("video") ? (
            <Video className="h-6 w-6" />
          ) : accept.includes("audio") ? (
            <Mic className="h-6 w-6" />
          ) : (
            <ImagePlus className="h-6 w-6" />
          )}
        </div>
      <div className="mt-4 text-base font-semibold text-slate-900">{title}</div>
      <div className="mt-1 text-sm text-slate-600">Or browse a file from your device.</div>
      {note ? <div className="mt-3 text-xs text-slate-500">{note}</div> : null}
      <div className="mt-6 flex justify-center">
        <Button type="button" variant="outline" onClick={openPicker}>
          Browse file
        </Button>
      </div>
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0]
          if (file) onFileSelected(file)
          e.currentTarget.value = ""
        }}
      />
    </div>
  )
}

