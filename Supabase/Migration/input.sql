-- Complete Database Setup Script with Vector Search Functions (FIXED)
-- This script sets up a complete hybrid search system with both functions

-- Create extension and table (unchanged)
CREATE EXTENSION IF NOT EXISTS vector;


-- clear chat history  
  -- Delete all rows only if any exist
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM n8n_chat_histories) THEN
        DELETE FROM n8n_chat_histories;
    END IF;
END $$;

  
-- Clean up: Drop everything related to old table names
DROP FUNCTION IF EXISTS hybrid_search(TEXT, VECTOR(512), INT, FLOAT, FLOAT, INT);
DROP FUNCTION IF EXISTS search_documents_hybrid(TEXT, VECTOR(1536), INT, JSONB, FLOAT, FLOAT, INT, TEXT);
DROP FUNCTION IF EXISTS search_similar_documents(VECTOR(1536), INT, JSONB, TEXT);
DROP TABLE IF EXISTS hybrid_search_vector_store_table CASCADE;
DROP TABLE IF EXISTS documents CASCADE;
DROP TABLE IF EXISTS document_embeddings CASCADE;
DROP TABLE IF EXISTS metadata_schema CASCADE;

-- Drop indexes if they exist (in case they weren't dropped with CASCADE)
DROP INDEX IF EXISTS hybrid_search_vector_store_table_fts_idx;
DROP INDEX IF EXISTS hybrid_search_vector_store_table_embedding_idx;
DROP INDEX IF EXISTS documents_fts_idx;
DROP INDEX IF EXISTS documents_embedding_idx;
DROP INDEX IF EXISTS documents_metadata_idx;

-- Drop extensions if they exist
DROP EXTENSION IF EXISTS vector CASCADE;
DROP EXTENSION IF EXISTS pg_trgm CASCADE;

-- Drop the old record manager table
DROP TABLE IF EXISTS record_manager;
DROP TABLE IF EXISTS document_records;
DROP TABLE IF EXISTS raw_data_table;
DROP TABLE IF EXISTS processed_files; 
DROP TABLE IF EXISTS deleted_files ; 
DROP TABLE IF EXISTS token_tracker; 

-- Recreate vector extension
CREATE EXTENSION IF NOT EXISTS vector;

create table token_tracker (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz default now(),
  file_id text not null , 

  -- token tracking
  prompt_tokens bigint default 0,
  completion_tokens bigint default 0,
  total_tokens bigint default 0,

  -- content tracking
  provided_content text,  -- user input / prompt
  ai_response text ,         -- AI’s output
  enhanced_chunk text -- combined reponse 
);


CREATE TABLE processed_files (
    file_id      text not null,
    file_title   TEXT NOT NULL,
    file_summary TEXT,
    file_type    TEXT,
    processed_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE deleted_files (
    file_id      text not null,
    file_title   TEXT NOT NULL,
    file_type    TEXT,
    deleted_at   TIMESTAMPTZ DEFAULT now()
);


-- Create document records table (renamed from record_manager)
CREATE TABLE document_records (
    id uuid primary key default gen_random_uuid(),
    doc_id text not null,
    file_name text  not null , 
    type text  not null  , 
    content_hash text not null,
    schema text ,
    created_at timestamp with time zone default timezone('utc', now())
);


CREATE TABLE raw_data_table (
    id SERIAL PRIMARY KEY,
    created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    doc_id  text not null,
    raw_data JSONB NOT NULL
);

-- Create metadata schema table
CREATE TABLE metadata_schema (
    id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    name TEXT NOT NULL UNIQUE,
    allowed_values TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::text, NOW()) NOT NULL
);

-- Insert efficient metadata fields for RAG
INSERT INTO metadata_schema (name, allowed_values) 
VALUES 
    ('department', 'HR,Finance,Engineering,Marketing,Sales,IT,Legal,Operations'),
    ('file_date', 'YYYY-MM-DD');


-- Create indexes for common query patterns
CREATE INDEX idx_metadata_schema_name ON metadata_schema(name);
CREATE INDEX idx_metadata_schema_created_at ON metadata_schema(created_at);

-- Create the main documents table (renamed from hybrid_search_vector_store_table)
CREATE TABLE documents (
    id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    content TEXT,
    fts TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    embedding VECTOR(1536),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::text, NOW()) NOT NULL
);

-- Create indexes with cleaner names
CREATE INDEX documents_fts_idx ON documents USING gin(fts);
CREATE INDEX documents_embedding_idx ON documents USING hnsw (embedding vector_ip_ops);
CREATE INDEX documents_metadata_idx ON documents USING gin(metadata);

-- Enhanced hybrid search function with FIXED metadata filtering for new JSON schema
CREATE OR REPLACE FUNCTION search_documents_hybrid(
    query_text text,
    query_embedding vector(1536),
    match_count int,
    metadata_filter jsonb DEFAULT '{}'::jsonb,
    full_text_weight float DEFAULT 1,
    semantic_weight float DEFAULT 1,
    rrf_k int DEFAULT 50,
    distance_method text DEFAULT 'cosine'
)
RETURNS TABLE (
    id bigint,
    content text,
    embedding vector(1536),
    metadata jsonb,
    created_at timestamp with time zone,
    -- Keyword/Full-text search results
    keyword_score double precision,
    keyword_rank int,
    -- Semantic search results
    similarity_score double precision,
    semantic_rank int,
    -- Final combined score
    rrf_score double precision
)
LANGUAGE plpgsql AS $$
DECLARE
    filter_conditions text := '';
    condition_text text;
    field_path text;
    operator_symbol text;
    cast_type text;
    condition_value text;
    group_condition text;
    final_where_clause text;
    filter_item jsonb;
    nested_filters jsonb;
    i int;
    j int;
BEGIN
    -- Validate distance method parameter
    IF distance_method NOT IN ('cosine', 'euclidean', 'inner_product') THEN
        RAISE EXCEPTION 'Invalid distance method: %. Supported methods are: cosine, euclidean, inner_product', distance_method;
    END IF;

    -- Build metadata filter conditions for new schema
    IF metadata_filter != '{}'::jsonb THEN
        -- Handle new "filters" array structure
        IF metadata_filter ? 'filters' THEN
            FOR i IN 0..jsonb_array_length(metadata_filter->'filters') - 1 LOOP
                filter_item := metadata_filter->'filters'->i;
                
                -- Handle nested AND conditions
                IF filter_item ? 'and' THEN
                    group_condition := '';
                    nested_filters := filter_item->'and';
                    
                    FOR j IN 0..jsonb_array_length(nested_filters) - 1 LOOP
                        -- Process each AND condition
                        field_path := 'metadata->>''' || (nested_filters->j->>'field') || '''';
                        
                        -- Map operators from new schema to PostgreSQL
                        operator_symbol := CASE nested_filters->j->>'operator'
                            WHEN 'eq' THEN '='
                            WHEN 'neq' THEN '!='
                            WHEN 'gt' THEN '>'
                            WHEN 'gte' THEN '>='
                            WHEN 'lt' THEN '<'
                            WHEN 'lte' THEN '<='
                            WHEN 'like' THEN 'LIKE'
                            WHEN 'ilike' THEN 'ILIKE'
                            WHEN 'is' THEN 'IS'
                            WHEN 'not' THEN 'IS NOT'
                            WHEN 'in' THEN 'IN'
                            WHEN 'contains' THEN '@>'
                            WHEN 'fts' THEN '@@'
                            WHEN 'match' THEN '~'
                            ELSE '='
                        END;
                        
                        -- Handle different value types and operators
                        IF nested_filters->j->>'operator' = 'in' THEN
                            -- Handle IN operator with array values
                            condition_value := '(' || string_agg(
                                CASE 
                                    WHEN jsonb_typeof(value) = 'string' THEN '''' || replace(value#>>'{}', '''', '''''') || ''''
                                    ELSE value#>>'{}'
                                END, 
                                ', '
                            ) || ')'
                            FROM jsonb_array_elements(nested_filters->j->'value') AS value;
                        ELSIF nested_filters->j->>'operator' = 'contains' THEN
                            -- Handle JSONB contains operator
                            field_path := 'metadata';
                            condition_value := '''' || replace(nested_filters->j->>'value', '''', '''''') || '''::jsonb';
                        ELSIF nested_filters->j->>'operator' = 'fts' THEN
                            -- Handle full-text search
                            field_path := 'fts';
                            condition_value := 'websearch_to_tsquery(''' || replace(nested_filters->j->>'value', '''', '''''') || ''')';
                        ELSIF jsonb_typeof(nested_filters->j->'value') = 'string' THEN
                            condition_value := '''' || replace(nested_filters->j->>'value', '''', '''''') || '''';
                        ELSIF nested_filters->j->'value' = 'null'::jsonb THEN
                            condition_value := 'NULL';
                        ELSE
                            condition_value := nested_filters->j->>'value';
                        END IF;
                        
                        condition_text := field_path || ' ' || operator_symbol || ' ' || condition_value;
                        
                        IF group_condition = '' THEN
                            group_condition := condition_text;
                        ELSE
                            group_condition := group_condition || ' AND ' || condition_text;
                        END IF;
                    END LOOP;
                    
                    -- Add AND group to main filter
                    IF group_condition != '' THEN
                        group_condition := '(' || group_condition || ')';
                        IF filter_conditions = '' THEN
                            filter_conditions := group_condition;
                        ELSE
                            filter_conditions := filter_conditions || ' AND ' || group_condition;
                        END IF;
                    END IF;
                    
                -- Handle nested OR conditions  
                ELSIF filter_item ? 'or' THEN
                    group_condition := '';
                    nested_filters := filter_item->'or';
                    
                    FOR j IN 0..jsonb_array_length(nested_filters) - 1 LOOP
                        -- Process each OR condition (similar to AND above)
                        field_path := 'metadata->>''' || (nested_filters->j->>'field') || '''';
                        
                        operator_symbol := CASE nested_filters->j->>'operator'
                            WHEN 'eq' THEN '='
                            WHEN 'neq' THEN '!='
                            WHEN 'gt' THEN '>'
                            WHEN 'gte' THEN '>='
                            WHEN 'lt' THEN '<'
                            WHEN 'lte' THEN '<='
                            WHEN 'like' THEN 'LIKE'
                            WHEN 'ilike' THEN 'ILIKE'
                            WHEN 'is' THEN 'IS'
                            WHEN 'not' THEN 'IS NOT'
                            WHEN 'in' THEN 'IN'
                            WHEN 'contains' THEN '@>'
                            WHEN 'fts' THEN '@@'
                            WHEN 'match' THEN '~'
                            ELSE '='
                        END;
                        
                        IF nested_filters->j->>'operator' = 'in' THEN
                            condition_value := '(' || string_agg(
                                CASE 
                                    WHEN jsonb_typeof(value) = 'string' THEN '''' || replace(value#>>'{}', '''', '''''') || ''''
                                    ELSE value#>>'{}'
                                END, 
                                ', '
                            ) || ')'
                            FROM jsonb_array_elements(nested_filters->j->'value') AS value;
                        ELSIF nested_filters->j->>'operator' = 'contains' THEN
                            field_path := 'metadata';
                            condition_value := '''' || replace(nested_filters->j->>'value', '''', '''''') || '''::jsonb';
                        ELSIF nested_filters->j->>'operator' = 'fts' THEN
                            field_path := 'fts';
                            condition_value := 'websearch_to_tsquery(''' || replace(nested_filters->j->>'value', '''', '''''') || ''')';
                        ELSIF jsonb_typeof(nested_filters->j->'value') = 'string' THEN
                            condition_value := '''' || replace(nested_filters->j->>'value', '''', '''''') || '''';
                        ELSIF nested_filters->j->'value' = 'null'::jsonb THEN
                            condition_value := 'NULL';
                        ELSE
                            condition_value := nested_filters->j->>'value';
                        END IF;
                        
                        condition_text := field_path || ' ' || operator_symbol || ' ' || condition_value;
                        
                        IF group_condition = '' THEN
                            group_condition := condition_text;
                        ELSE
                            group_condition := group_condition || ' OR ' || condition_text;
                        END IF;
                    END LOOP;
                    
                    -- Add OR group to main filter
                    IF group_condition != '' THEN
                        group_condition := '(' || group_condition || ')';
                        IF filter_conditions = '' THEN
                            filter_conditions := group_condition;
                        ELSE
                            filter_conditions := filter_conditions || ' AND ' || group_condition;
                        END IF;
                    END IF;
                    
                -- Handle simple direct conditions
                ELSIF filter_item ? 'field' AND filter_item ? 'operator' THEN
                    field_path := 'metadata->>''' || (filter_item->>'field') || '''';
                    
                    operator_symbol := CASE filter_item->>'operator'
                        WHEN 'eq' THEN '='
                        WHEN 'neq' THEN '!='
                        WHEN 'gt' THEN '>'
                        WHEN 'gte' THEN '>='
                        WHEN 'lt' THEN '<'
                        WHEN 'lte' THEN '<='
                        WHEN 'like' THEN 'LIKE'
                        WHEN 'ilike' THEN 'ILIKE'
                        WHEN 'is' THEN 'IS'
                        WHEN 'not' THEN 'IS NOT'
                        WHEN 'in' THEN 'IN'
                        WHEN 'contains' THEN '@>'
                        WHEN 'fts' THEN '@@'
                        WHEN 'match' THEN '~'
                        ELSE '='
                    END;
                    
                    -- Handle different value types
                    IF filter_item->>'operator' = 'in' THEN
                        condition_value := '(' || string_agg(
                            CASE 
                                WHEN jsonb_typeof(value) = 'string' THEN '''' || replace(value#>>'{}', '''', '''''') || ''''
                                ELSE value#>>'{}'
                            END, 
                            ', '
                        ) || ')'
                        FROM jsonb_array_elements(filter_item->'value') AS value;
                    ELSIF filter_item->>'operator' = 'contains' THEN
                        field_path := 'metadata';
                        IF jsonb_typeof(filter_item->'value') = 'array' THEN
                            condition_value := '''' || filter_item->'value'::text || '''::jsonb';
                        ELSE
                            condition_value := '''' || replace(filter_item->>'value', '''', '''''') || '''::jsonb';
                        END IF;
                    ELSIF filter_item->>'operator' = 'fts' THEN
                        field_path := 'fts';
                        condition_value := 'websearch_to_tsquery(''' || replace(filter_item->>'value', '''', '''''') || ''')';
                    ELSIF jsonb_typeof(filter_item->'value') = 'string' THEN
                        condition_value := '''' || replace(filter_item->>'value', '''', '''''') || '''';
                    ELSIF filter_item->'value' = 'null'::jsonb THEN
                        condition_value := 'NULL';
                    ELSE
                        condition_value := filter_item->>'value';
                    END IF;
                    
                    condition_text := field_path || ' ' || operator_symbol || ' ' || condition_value;
                    
                    IF filter_conditions = '' THEN
                        filter_conditions := condition_text;
                    ELSE
                        filter_conditions := filter_conditions || ' AND ' || condition_text;
                    END IF;
                END IF;
            END LOOP;
        END IF;
        
        -- Handle legacy format (fallback for backward compatibility)
        IF NOT (metadata_filter ? 'filters') AND metadata_filter != '{}'::jsonb THEN
            filter_conditions := 'metadata @> ''' || metadata_filter::text || '''::jsonb';
        END IF;
    END IF;
    
    -- Build final WHERE clause
    IF filter_conditions = '' THEN
        final_where_clause := 'true';
    ELSE
        final_where_clause := filter_conditions;
    END IF;

    RETURN QUERY EXECUTE format('
    WITH filtered_records AS (
        SELECT * FROM documents
        WHERE %s
    ),
    full_text AS (
        SELECT 
            fr.id,
            row_number() OVER(ORDER BY ts_rank_cd(fr.fts, websearch_to_tsquery($1)) DESC)::int as rank_ix,
            ts_rank_cd(fr.fts, websearch_to_tsquery($1))::double precision as keyword_score
        FROM filtered_records fr
        WHERE fr.fts @@ websearch_to_tsquery($1)
        ORDER BY keyword_score DESC
        LIMIT LEAST($3, 30) * 2
    ),
    semantic AS (
        SELECT 
            fr.id,
            row_number() OVER (
                ORDER BY CASE $8
                    WHEN ''cosine'' THEN fr.embedding <=> $2
                    WHEN ''euclidean'' THEN fr.embedding <-> $2
                    WHEN ''inner_product'' THEN -(fr.embedding <#> $2)
                END
            )::int as rank_ix,
            (CASE $8
                WHEN ''cosine'' THEN 1 - (fr.embedding <=> $2)
                WHEN ''euclidean'' THEN 1 / (1 + (fr.embedding <-> $2))
                WHEN ''inner_product'' THEN fr.embedding <#> $2
            END)::double precision as similarity_score
        FROM filtered_records fr
        ORDER BY CASE $8
            WHEN ''cosine'' THEN fr.embedding <=> $2
            WHEN ''euclidean'' THEN fr.embedding <-> $2
            WHEN ''inner_product'' THEN -(fr.embedding <#> $2)
        END
        LIMIT LEAST($3, 30) * 2
    ),
    combined_results AS (
        SELECT 
            COALESCE(ft.id, s.id) as id,
            ft.keyword_score,
            ft.rank_ix as keyword_rank,
            s.similarity_score,
            s.rank_ix as semantic_rank,
            (COALESCE(1.0 / ($7 + ft.rank_ix), 0.0) * $5 + 
             COALESCE(1.0 / ($7 + s.rank_ix), 0.0) * $6)::double precision as rrf_score
        FROM full_text ft
        FULL OUTER JOIN semantic s ON ft.id = s.id
    )
    SELECT 
        vst.id,
        vst.content,
        vst.embedding,
        vst.metadata,
        vst.created_at,
        cr.keyword_score,
        cr.keyword_rank,
        cr.similarity_score,
        cr.semantic_rank,
        cr.rrf_score
    FROM combined_results cr
    JOIN documents vst ON cr.id = vst.id
    ORDER BY cr.rrf_score DESC
    LIMIT LEAST($3, 30)', final_where_clause)
    USING query_text, query_embedding, match_count, metadata_filter, full_text_weight, semantic_weight, rrf_k, distance_method;
END;
$$;
-- Fixed similarity search function with proper type casting
CREATE OR REPLACE FUNCTION search_similar_documents(
    query_embedding VECTOR(1536),
    match_count INT DEFAULT 10,
    metadata_filter JSONB DEFAULT '{}'::jsonb,
    distance_method TEXT DEFAULT 'cosine'
)
RETURNS TABLE (
    id BIGINT,
    content TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE,
    similarity double precision  -- Changed from FLOAT to double precision
)
LANGUAGE plpgsql AS $$
BEGIN
    -- Validate distance method parameter
    IF distance_method NOT IN ('cosine', 'euclidean', 'inner_product') THEN
        RAISE EXCEPTION 'Invalid distance method: %. Supported methods are: cosine, euclidean, inner_product', distance_method;
    END IF;

    -- Execute query based on selected distance method
    CASE distance_method
        WHEN 'cosine' THEN
            RETURN QUERY
            SELECT 
                d.id,
                d.content,
                d.metadata,
                d.created_at,
                (1 - (d.embedding <=> query_embedding))::double precision AS similarity  -- FIXED: Cast to double precision
            FROM documents d
            WHERE d.metadata @> metadata_filter
            ORDER BY d.embedding <=> query_embedding
            LIMIT match_count;
            
        WHEN 'euclidean' THEN
            RETURN QUERY
            SELECT 
                d.id,
                d.content,
                d.metadata,
                d.created_at,
                (1.0 / (1.0 + (d.embedding <-> query_embedding)))::double precision AS similarity  -- FIXED: Cast to double precision
            FROM documents d
            WHERE d.metadata @> metadata_filter
            ORDER BY d.embedding <-> query_embedding
            LIMIT match_count;
            
        WHEN 'inner_product' THEN
            RETURN QUERY
            SELECT 
                d.id,
                d.content,
                d.metadata,
                d.created_at,
                (-(d.embedding <#> query_embedding))::double precision AS similarity  -- FIXED: Cast to double precision
            FROM documents d
            WHERE d.metadata @> metadata_filter
            ORDER BY d.embedding <#> query_embedding DESC
            LIMIT match_count;
    END CASE;
END;
$$;
