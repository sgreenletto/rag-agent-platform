CREATE DATABASE IF NOT EXISTS rag_agent
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE rag_agent;

CREATE TABLE IF NOT EXISTS documents (
    document_id VARCHAR(191) PRIMARY KEY,
    filename VARCHAR(512) NOT NULL,
    file_type VARCHAR(64) NOT NULL,
    source_path TEXT NOT NULL,
    status VARCHAR(64) NOT NULL,
    metadata JSON NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_documents_filename (filename),
    INDEX idx_documents_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS parent_chunks (
    chunk_id VARCHAR(191) PRIMARY KEY,
    document_id VARCHAR(191) NOT NULL,
    content LONGTEXT NOT NULL,
    page INT NULL,
    metadata JSON NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_parent_chunks_document_id (document_id),
    CONSTRAINT fk_parent_chunks_document
        FOREIGN KEY (document_id)
        REFERENCES documents(document_id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS child_chunks (
    chunk_id VARCHAR(191) PRIMARY KEY,
    document_id VARCHAR(191) NOT NULL,
    parent_id VARCHAR(191) NOT NULL,
    content LONGTEXT NOT NULL,
    page INT NULL,
    metadata JSON NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_child_chunks_document_id (document_id),
    INDEX idx_child_chunks_parent_id (parent_id),
    CONSTRAINT fk_child_chunks_document
        FOREIGN KEY (document_id)
        REFERENCES documents(document_id)
        ON DELETE CASCADE,
    CONSTRAINT fk_child_chunks_parent
        FOREIGN KEY (parent_id)
        REFERENCES parent_chunks(chunk_id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
