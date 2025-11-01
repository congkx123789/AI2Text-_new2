-- Metadata Service Database Schema
-- PostgreSQL migration for ACID metadata store

CREATE TYPE split_enum AS ENUM ('TRAIN','VAL','TEST');

CREATE TABLE speakers (
  speaker_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  pseudonymous_id TEXT UNIQUE NOT NULL,  -- Never expose PII
  region TEXT,              -- optional: north/central/south
  device_types TEXT[],       -- array of device types
  total_recordings INTEGER DEFAULT 0,
  created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE audio (
  audio_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  speaker_id UUID REFERENCES speakers(speaker_id) ON DELETE SET NULL,
  audio_path TEXT NOT NULL,              -- s3://... or minio path
  snr_estimate REAL,                     -- for drift & stratification
  device_type TEXT,                      -- mic/headset/mobile
  environment TEXT,                      -- quiet/office/street...
  split_assignment split_enum NOT NULL,  -- enforced by trigger below
  duration_seconds REAL,
  sample_rate INTEGER,
  created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE transcripts (
  audio_id UUID PRIMARY KEY REFERENCES audio(audio_id) ON DELETE CASCADE,
  raw_json JSONB,     -- ASR lattice or best path + word timings
  text TEXT,          -- normalized text (post ASR)
  text_clean TEXT,    -- after nlp-post (diacritics, typos)
  created_at TIMESTAMP DEFAULT now(),
  updated_at TIMESTAMP DEFAULT now()
);

-- Speaker-level split guard: a speaker must not appear in multiple splits
CREATE FUNCTION enforce_speaker_split() RETURNS trigger AS $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM audio a
    WHERE a.speaker_id = NEW.speaker_id
      AND a.split_assignment <> NEW.split_assignment
  ) THEN
    RAISE EXCEPTION 'Speaker % already assigned to a different split', NEW.speaker_id;
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_speaker_split
BEFORE INSERT OR UPDATE ON audio
FOR EACH ROW EXECUTE FUNCTION enforce_speaker_split();

-- Indexes for performance
CREATE INDEX idx_audio_speaker ON audio(speaker_id);
CREATE INDEX idx_audio_snr ON audio(snr_estimate);
CREATE INDEX idx_audio_device ON audio(device_type);
CREATE INDEX idx_audio_split ON audio(split_assignment);
CREATE INDEX idx_speakers_pseudonymous ON speakers(pseudonymous_id);

-- Update timestamp trigger
CREATE FUNCTION update_updated_at() RETURNS trigger AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_transcripts_updated_at
BEFORE UPDATE ON transcripts
FOR EACH ROW EXECUTE FUNCTION update_updated_at();

