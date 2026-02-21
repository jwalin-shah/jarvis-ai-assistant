const iterations = 1000;
const arraySize = 1000;

// Generate random ISO dates
const generateDate = () => new Date(Date.now() - Math.floor(Math.random() * 10000000000)).toISOString();
const dates = Array.from({ length: arraySize }, () => ({ last_message_date: generateDate() }));

console.log(`Benchmarking sort of ${arraySize} items over ${iterations} iterations...`);

// Method 1: Date parsing
const start1 = performance.now();
for (let i = 0; i < iterations; i++) {
  const arr = [...dates];
  arr.sort((a, b) => new Date(b.last_message_date).getTime() - new Date(a.last_message_date).getTime());
}
const end1 = performance.now();
console.log(`Method 1 (new Date): ${(end1 - start1).toFixed(2)}ms`);

// Method 2: String comparison
const start2 = performance.now();
for (let i = 0; i < iterations; i++) {
  const arr = [...dates];
  arr.sort((a, b) => {
    if (b.last_message_date > a.last_message_date) return 1;
    if (b.last_message_date < a.last_message_date) return -1;
    return 0;
  });
}
const end2 = performance.now();
console.log(`Method 2 (String compare): ${(end2 - start2).toFixed(2)}ms`);

// Method 3: localeCompare
const start3 = performance.now();
for (let i = 0; i < iterations; i++) {
  const arr = [...dates];
  arr.sort((a, b) => b.last_message_date.localeCompare(a.last_message_date));
}
const end3 = performance.now();
console.log(`Method 3 (localeCompare): ${(end3 - start3).toFixed(2)}ms`);
