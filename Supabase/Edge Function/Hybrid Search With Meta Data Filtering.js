import { createClient } from 'npm:@supabase/supabase-js@2';
import OpenAI from 'npm:openai';
const supabaseUrl = Deno.env.get('SUPABASE_URL');
const supabaseServiceRoleKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY');
const openaiApiKey = Deno.env.get('OPEN_API_KEY');
// Validation function for the new filter schema
function validateFilterSchema(filter) {
  if (!filter || typeof filter !== 'object') return true;
  if (filter.filters && Array.isArray(filter.filters)) {
    for (const filterItem of filter.filters){
      // Check if it's a simple filter
      if (filterItem.field && filterItem.operator) {
        const validOperators = [
          'eq',
          'neq',
          'gt',
          'gte',
          'lt',
          'lte',
          'like',
          'ilike',
          'in',
          'contains',
          'is',
          'not',
          'fts',
          'match'
        ];
        if (!validOperators.includes(filterItem.operator)) {
          return false;
        }
      } else if (filterItem.and || filterItem.or) {
        const nestedFilters = filterItem.and || filterItem.or;
        if (!Array.isArray(nestedFilters)) return false;
        for (const nested of nestedFilters){
          if (!nested.field || !nested.operator) return false;
          const validOperators = [
            'eq',
            'neq',
            'gt',
            'gte',
            'lt',
            'lte',
            'like',
            'ilike',
            'in',
            'contains',
            'is',
            'not',
            'fts',
            'match'
          ];
          if (!validOperators.includes(nested.operator)) return false;
        }
      } else {
        return false; // Invalid filter structure
      }
    }
  }
  return true;
}
// Transform the new filter schema to the format expected by the PostgreSQL function
function transformFilterSchema(inputFilter) {
  // If it's empty or already in the correct format, return as-is
  if (!inputFilter || !inputFilter.filters) {
    return inputFilter || {};
  }
  // Handle the case where we receive the nested "filter" object
  const filtersArray = inputFilter.filter?.filters || inputFilter.filters;
  if (!Array.isArray(filtersArray)) {
    return inputFilter;
  }
  // Transform to the expected format
  return {
    filters: filtersArray
  };
}
Deno.serve(async (req)=>{
  try {
    // Parse the request body
    const requestBody = await req.json();
    // Handle both direct parameters and nested filter object
    let { query, metadata_filter = {}, match_count = 10, full_text_weight = 1.0, semantic_weight = 1.0, rrf_k = 50, distance_method = 'cosine' } = requestBody;
    // If filter is nested in a "filter" object, extract it
    if (requestBody.filter) {
      metadata_filter = requestBody.filter;
    }
    if (!query) {
      return new Response(JSON.stringify({
        error: 'Query parameter is required'
      }), {
        status: 400,
        headers: {
          'Content-Type': 'application/json'
        }
      });
    }
    // Validate distance_method
    const validDistanceMethods = [
      'cosine',
      'euclidean',
      'inner_product'
    ];
    if (!validDistanceMethods.includes(distance_method)) {
      return new Response(JSON.stringify({
        error: `Invalid distance method: ${distance_method}. Supported methods are: ${validDistanceMethods.join(', ')}`
      }), {
        status: 400,
        headers: {
          'Content-Type': 'application/json'
        }
      });
    }
    // Validate filter schema
    if (!validateFilterSchema(metadata_filter)) {
      return new Response(JSON.stringify({
        error: 'Invalid filter schema. Please check the filter structure and operators.'
      }), {
        status: 400,
        headers: {
          'Content-Type': 'application/json'
        }
      });
    }
    // Transform filter schema to the expected format
    const transformedFilter = transformFilterSchema(metadata_filter);
    // Instantiate OpenAI client
    const openai = new OpenAI({
      apiKey: openaiApiKey
    });
    // Generate embedding for the user's query
    const embeddingResponse = await openai.embeddings.create({
      model: 'text-embedding-3-small',
      input: query,
      dimensions: 1536
    });
    const [{ embedding }] = embeddingResponse.data;
    // Instantiate the Supabase client
    const supabase = createClient(supabaseUrl, supabaseServiceRoleKey);
    // Call the enhanced hybrid_search function via RPC
    const { data: documents, error } = await supabase.rpc('search_documents_hybrid', {
      query_text: query,
      query_embedding: embedding,
      match_count: match_count,
      metadata_filter: transformedFilter,
      full_text_weight: full_text_weight,
      semantic_weight: semantic_weight,
      rrf_k: rrf_k,
      distance_method: distance_method
    });
    if (error) {
      console.error('Supabase RPC error:', error);
      return new Response(JSON.stringify({
        error: 'Search failed',
        details: error.message,
        debug_info: {
          original_filter: metadata_filter,
          transformed_filter: transformedFilter
        }
      }), {
        status: 500,
        headers: {
          'Content-Type': 'application/json'
        }
      });
    }
    // Transform results to include detailed scoring information
    const enhancedResults = documents.map((doc)=>({
        id: doc.id,
        content: doc.content,
        metadata: doc.metadata,
        created_at: doc.created_at,
        search_scores: {
          // Keyword/Full-text search results
          keyword_score: doc.keyword_score,
          keyword_rank: doc.keyword_rank,
          // Semantic search results
          similarity_score: doc.similarity_score,
          semantic_rank: doc.semantic_rank,
          // Final combined score
          rrf_score: doc.rrf_score
        }
      }));
    return new Response(JSON.stringify({
      results: enhancedResults,
      search_params: {
        query,
        original_filter: metadata_filter,
        transformed_filter: transformedFilter,
        match_count,
        full_text_weight,
        semantic_weight,
        rrf_k,
        distance_method
      },
      summary: {
        total_results: enhancedResults.length,
        has_keyword_matches: enhancedResults.some((r)=>r.search_scores.keyword_score !== null),
        has_semantic_matches: enhancedResults.some((r)=>r.search_scores.similarity_score !== null),
        distance_method_used: distance_method,
        filter_applied: Object.keys(transformedFilter).length > 0
      }
    }), {
      headers: {
        'Content-Type': 'application/json'
      }
    });
  } catch (error) {
    console.error('Edge function error:', error);
    return new Response(JSON.stringify({
      error: 'Internal server error',
      details: error.message,
      stack: error.stack
    }), {
      status: 500,
      headers: {
        'Content-Type': 'application/json'
      }
    });
  }
});
