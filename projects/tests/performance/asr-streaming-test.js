import ws from 'k6/ws';
import { check } from 'k6';
import { Rate, Trend } from 'k6/metrics';

const errorRate = new Rate('streaming_errors');
const e2eLatency = new Trend('e2e_latency');
const partialLatency = new Trend('partial_latency');

export const options = {
  stages: [
    { duration: '1m', target: 10 },   // Ramp to 10 connections
    { duration: '3m', target: 50 },   // Ramp to 50 (target)
    { duration: '5m', target: 50 },   // Sustain 50 connections
    { duration: '1m', target: 0 },    // Ramp down
  ],
  thresholds: {
    'e2e_latency': ['p(95)<500'],         // p95 < 500ms E2E
    'partial_latency': ['p(95)<300'],     // p95 < 300ms for partials
    'streaming_errors': ['rate<0.001'],   // < 0.1% error rate
  },
};

const WS_URL = __ENV.WS_URL || 'ws://localhost:8080/stream';

export default function () {
  const res = ws.connect(WS_URL, {}, function (socket) {
    socket.on('open', () => {
      console.log('Connected to ASR streaming');
      
      // Simulate audio chunks
      const chunkCount = 50;
      const chunkSize = 4096;
      
      for (let i = 0; i < chunkCount; i++) {
        const audioChunk = new Uint8Array(chunkSize);
        const startTime = Date.now();
        
        socket.send(audioChunk.buffer);
        
        socket.on('message', (data) => {
          const endTime = Date.now();
          const latency = endTime - startTime;
          
          const message = JSON.parse(data);
          
          if (message.is_final) {
            e2eLatency.add(latency);
          } else {
            partialLatency.add(latency);
          }
          
          check(message, {
            'has text': (m) => m.text !== undefined,
            'latency acceptable': () => latency < 500,
          });
        });
        
        // Sleep to simulate real-time audio rate (~100ms per chunk)
        sleep(0.1);
      }
    });
    
    socket.on('error', (e) => {
      console.log('WebSocket error:', e);
      errorRate.add(1);
    });
    
    socket.on('close', () => {
      console.log('Disconnected from ASR streaming');
    });
    
    socket.setTimeout(() => {
      socket.close();
    }, 60000);  // 60 second timeout
  });
  
  check(res, {
    'status is 101': (r) => r && r.status === 101,
  });
}

export function handleSummary(data) {
  const e2eP95 = data.metrics.e2e_latency?.values['p(95)'] || 0;
  const partialP95 = data.metrics.partial_latency?.values['p(95)'] || 0;
  
  console.log('\n========== ASR STREAMING PERFORMANCE ==========');
  console.log(`E2E p95: ${e2eP95.toFixed(2)}ms (target: < 500ms)`);
  console.log(`Partial p95: ${partialP95.toFixed(2)}ms (target: < 300ms)`);
  console.log(`Error rate: ${(data.metrics.streaming_errors?.values.rate || 0) * 100}%`);
  
  const sloMet = e2eP95 < 500 && data.metrics.streaming_errors?.values.rate < 0.001;
  console.log(`\nSLO Status: ${sloMet ? '✅ PASSED' : '❌ FAILED'}\n`);
  
  return {
    'results/asr-streaming-summary.json': JSON.stringify(data),
  };
}

