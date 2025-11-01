import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

const errorRate = new Rate('search_errors');
const searchLatency = new Trend('search_latency');

export const options = {
  stages: [
    { duration: '1m', target: 50 },
    { duration: '5m', target: 50 },
    { duration: '1m', target: 100 },
    { duration: '2m', target: 100 },
    { duration: '1m', target: 0 },
  ],
  thresholds: {
    'http_req_duration': ['p(95)<50', 'p(99)<120'],  // p95 < 50ms, p99 < 120ms
    'http_req_failed': ['rate<0.01'],
    'search_errors': ['rate<0.01'],
  },
};

const BASE_URL = __ENV.SEARCH_URL || 'http://localhost:8080';
const QUERIES = [
  'machine learning',
  'neural network',
  'deep learning',
  'artificial intelligence',
  'computer vision',
  'natural language processing',
  'speech recognition',
  'Vietnamese language',
];

export default function () {
  const query = QUERIES[Math.floor(Math.random() * QUERIES.length)];
  
  const res = http.get(`${BASE_URL}/search?q=${encodeURIComponent(query)}&limit=10&threshold=0.7`);
  
  const success = check(res, {
    'status is 200': (r) => r.status === 200,
    'has results': (r) => JSON.parse(r.body).total > 0,
    'p95 latency < 50ms': (r) => r.timings.duration < 50,
    'p99 latency < 120ms': (r) => r.timings.duration < 120,
    'query time tracked': (r) => JSON.parse(r.body).query_time_ms !== undefined,
  });
  
  if (res.status === 200) {
    const body = JSON.parse(res.body);
    searchLatency.add(body.query_time_ms);
  }
  
  errorRate.add(!success);
  
  sleep(Math.random() * 2 + 0.5);  // Random sleep 0.5-2.5s
}

export function handleSummary(data) {
  const p95 = data.metrics.http_req_duration.values['p(95)'];
  const p99 = data.metrics.http_req_duration.values['p(99)'];
  
  console.log('\n========== SEARCH PERFORMANCE SUMMARY ==========');
  console.log(`p95 latency: ${p95.toFixed(2)}ms (target: < 50ms)`);
  console.log(`p99 latency: ${p99.toFixed(2)}ms (target: < 120ms)`);
  console.log(`Error rate: ${(data.metrics.http_req_failed.values.rate * 100).toFixed(2)}%`);
  console.log(`Total requests: ${data.metrics.http_reqs.values.count}`);
  
  const sloMet = p95 < 50 && p99 < 120 && data.metrics.http_req_failed.values.rate < 0.01;
  console.log(`\nSLO Status: ${sloMet ? '✅ PASSED' : '❌ FAILED'}\n`);
  
  return {
    'results/search-summary.json': JSON.stringify(data),
  };
}

