-- AI2Text Metadata Database Schema V1

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Recordings table
CREATE TABLE recordings (
    recording_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    status VARCHAR(50) NOT NULL,
    audio_url TEXT NOT NULL,
    transcript TEXT,
    language VARCHAR(10) NOT NULL,
    duration_sec FLOAT,
    error TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_recordings_status ON recordings(status);
CREATE INDEX idx_recordings_created_at ON recordings(created_at DESC);
CREATE INDEX idx_recordings_language ON recordings(language);
CREATE INDEX idx_recordings_metadata ON recordings USING gin(metadata);

-- Updated timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_recordings_updated_at BEFORE UPDATE
    ON recordings FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Comments
COMMENT ON TABLE recordings IS 'Audio recording metadata';
COMMENT ON COLUMN recordings.status IS 'uploaded, transcribing, transcribed, processing, completed, failed';
COMMENT ON COLUMN recordings.metadata IS 'Additional user-provided metadata as JSON';
