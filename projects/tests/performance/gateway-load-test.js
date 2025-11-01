import http from 'k6/http';
import { check, group, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const latency = new Trend('latency');

// Test configuration
export const options = {
  stages: [
    { duration: '2m', target: 100 },   // Ramp-up to 100 VUs
    { duration: '5m', target: 100 },   // Sustain 100 VUs
    { duration: '2m', target: 1000 },  // Spike to 1000 VUs
    { duration: '1m', target: 1000 },  // Hold spike
    { duration: '1m', target: 0 },     // Ramp-down
  ],
  thresholds: {
    'http_req_duration': ['p(95)<150'],  // p95 < 150ms
    'http_req_failed': ['rate<0.01'],    // < 1% errors
    'errors': ['rate<0.01'],             // < 1% errors
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8080';
const AUTH_TOKEN = __ENV.AUTH_TOKEN || 'dev-token';

export default function () {
  group('Health Check', () => {
    const res = http.get(`${BASE_URL}/health`);
    
    check(res, {
      'status is 200': (r) => r.status === 200,
      'response time < 100ms': (r) => r.timings.duration < 100,
    });
    
    errorRate.add(res.status !== 200);
    latency.add(res.timings.duration);
  });

  sleep(0.1);

  group('Search Request', () => {
    const params = {
      headers: {
        'Authorization': `Bearer ${AUTH_TOKEN}`,
        'Content-Type': 'application/json',
      },
    };
    
    const res = http.get(`${BASE_URL}/search?q=test&limit=10`, params);
    
    check(res, {
      'status is 200': (r) => r.status === 200,
      'has results': (r) => JSON.parse(r.body).hits.length > 0,
      'response time < 150ms': (r) => r.timings.duration < 150,
    });
    
    errorRate.add(res.status !== 200);
    latency.add(res.timings.duration);
  });

  sleep(0.5);

  group('Metadata Fetch', () => {
    const params = {
      headers: {
        'Authorization': `Bearer ${AUTH_TOKEN}`,
      },
    };
    
    const res = http.get(`${BASE_URL}/metadata/test-id`, params);
    
    check(res, {
      'status is 200 or 404': (r) => r.status === 200 || r.status === 404,
      'response time < 50ms': (r) => r.timings.duration < 50,
    });
    
    errorRate.add(res.status >= 500);
    latency.add(res.timings.duration);
  });

  sleep(1);
}

export function handleSummary(data) {
  return {
    'results/gateway-summary.json': JSON.stringify(data),
    stdout: textSummary(data, { indent: ' ', enableColors: true }),
  };
}

function textSummary(data, options) {
  const indent = options.indent || '';
  const enableColors = options.enableColors || false;
  
  let summary = '\n';
  summary += `${indent}✓ checks.........................: ${(data.metrics.checks.values.passes / data.metrics.checks.values.count * 100).toFixed(2)}%\n`;
  summary += `${indent}✓ http_req_duration..............: avg=${data.metrics.http_req_duration.values.avg.toFixed(2)}ms p(95)=${data.metrics.http_req_duration.values['p(95)'].toFixed(2)}ms\n`;
  summary += `${indent}✓ http_req_failed................: ${(data.metrics.http_req_failed.values.rate * 100).toFixed(2)}%\n`;
  summary += `${indent}✓ http_reqs......................: ${data.metrics.http_reqs.values.count} (${data.metrics.http_reqs.values.rate.toFixed(2)}/s)\n`;
  summary += `${indent}✓ vus............................: ${data.metrics.vus.values.max} max\n`;
  
  return summary;
}

