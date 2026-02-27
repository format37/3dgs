import { readFileSync, writeFileSync } from 'fs';
import { loadPly, serializeSpz } from 'spz-js';

const plyPath = process.argv[2] || 'export/splat.ply';
const spzPath = process.argv[3] || 'export/scene.spz';

console.log(`Loading PLY from ${plyPath}...`);
const plyData = readFileSync(plyPath);
const splat = await loadPly(new Blob([plyData]).stream());
console.log(`Loaded ${splat.numPoints} gaussians`);

console.log(`Serializing to SPZ...`);
const spzData = await serializeSpz(splat);
writeFileSync(spzPath, Buffer.from(spzData));

const plySize = (plyData.length / 1024 / 1024).toFixed(1);
const spzSize = (spzData.byteLength / 1024 / 1024).toFixed(1);
console.log(`Done: ${spzPath} (${spzSize} MB, was ${plySize} MB PLY)`);
