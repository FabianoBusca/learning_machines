import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterPlot, Scatter, AreaChart, Area, BarChart, Bar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';

export function App(props) {
  const [data, setData] = useState([]);
  const [selectedSensor, setSelectedSensor] = useState(0);

  // Sample data from the JSON file
  const jsonData = {
    "genotype": [0.5, 0.5, 0.5, 0.5, -1.0, -0.5, -1.0, -0.5, -1.2, -1.2, -1.0, -0.5, 0.5, 0.5, 0.0, 0.0],
    "steps": [
      {"timestamp": "2025-06-10T09:34:31.117802", "irs": [6.437948873615927, 6.44148393850715, 60.22741353587145, 60.323400944404334, 5.842482405981195, 8.12478706847045, 57.79294628375504, 5.922248379178085], "left_speed": -100, "right_speed": -100},
      {"timestamp": "2025-06-10T09:34:31.863055", "irs": [20.558858932887723, 22.0466112409676, 60.2535875263511, 60.34709077212538, 5.847945467659501, 8.17946307802341, 65.04587280004668, 5.926568185848894], "left_speed": -100, "right_speed": -100},
      {"timestamp": "2025-06-10T09:34:32.464805", "irs": [2009.3938272162436, 2696.8433010511244, 60.22942196465359, 60.32353150077513, 5.842951603604075, 5.889071091765259, 87409.27576151252, 5.922750530950787], "left_speed": -100, "right_speed": -100},
      {"timestamp": "2025-06-10T09:34:33.113342", "irs": [2064.539319123466, 4381.432703551845, 60.51611455980222, 60.626378394379415, 5.762795162502442, 5.8329578786545015, 2402005.8025241885, 5.864568611726452], "left_speed": -100, "right_speed": -100},
      {"timestamp": "2025-06-10T09:34:33.863632", "irs": [1654.5345007854046, 5810.717708931056, 60.2977584836963, 60.395071667689784, 5.833284040346645, 5.882306683458027, 2073461.4661700507, 5.915948832348836], "left_speed": -100, "right_speed": -100},
      {"timestamp": "2025-06-10T09:34:34.563378", "irs": [1273.7913295137057, 9013.544348864074, 60.2469511442462, 60.34112800676512, 5.815886109684514, 5.869980711333119, 2642730.2913324204, 5.9036699545977065], "left_speed": -100, "right_speed": -100},
      {"timestamp": "2025-06-10T09:34:35.212649", "irs": [950.8117015715329, 17160.4957120663, 64.04587668326079, 64.16605401919978, 6.394207306763671, 6.2797266212576766, 3318109.043120337, 6.305446665982607], "left_speed": -100, "right_speed": -100},
      {"timestamp": "2025-06-10T09:34:35.863572", "irs": [908.5209216140686, 15748.198045223508, 186.90958245321502, 189.2355369542491, 44.23982255915803, 27.003675565123768, 893842.2900376016, 26.73122414130563], "left_speed": -100, "right_speed": -100},
      {"timestamp": "2025-06-10T09:34:36.562751", "irs": [930.1144438180959, 17958.38635163247, 62.79796641909654, 62.91073525158424, 6.162383655461086, 6.115644388255361, 2914890.391587674, 6.145936793414384], "left_speed": -100, "right_speed": -100},
      {"timestamp": "2025-06-10T09:34:37.213522", "irs": [922.2462015847533, 18127.077456438423, 61.89791192109438, 62.01735371719367, 5.963988700663531, 5.9763260427933975, 2550147.137047537, 6.006335364404769], "left_speed": -100, "right_speed": -100},
      {"timestamp": "2025-06-10T09:34:37.864423", "irs": [923.7551378810156, 18041.247453402477, 61.686964006045976, 61.828968025550914, 5.893457390738095, 5.925564824661648, 2534540.755116306, 5.957803377886583], "left_speed": -100, "right_speed": -100},
      {"timestamp": "2025-06-10T09:34:38.562725", "irs": [935.2145120901399, 17717.931588295167, 60.08612297052816, 60.190776388337035, 5.810360741935277, 5.867350751980097, 2910030.882979718, 5.89776124471328], "left_speed": -100, "right_speed": -100},
      {"timestamp": "2025-06-10T09:34:39.114185", "irs": [912.0025542753617, 16002.446852328685, 179.11142396309265, 179.3990638641081, 40.09350094229868, 25.621632043053427, 1027315.9720149594, 25.42553271679617], "left_speed": -100, "right_speed": -100},
      {"timestamp": "2025-06-10T09:34:39.767266", "irs": [917.5931237879065, 17951.696050536466, 60.928144461029916, 61.0455055310173, 5.730057609340398, 5.809897110907765, 2059946.5359993603, 5.841890647322832], "left_speed": -100, "right_speed": -100},
      {"timestamp": "2025-06-10T09:34:40.364209", "irs": [906.3066739125724, 17926.436632220593, 60.66966017741349, 60.7879950748837, 5.694251738225026, 5.785563414679746, 1573935.0056859925, 5.81474801171987], "left_speed": -100, "right_speed": -100},
      {"timestamp": "2025-06-10T09:34:41.013066", "irs": [922.8725273469479, 18097.738294445753, 62.014339342075594, 62.09418248758324, 5.958494513871451, 5.9713678603810605, 2553808.9473067513, 6.003858835540498], "left_speed": -100, "right_speed": -100},
      {"timestamp": "2025-06-10T09:34:41.618877", "irs": [921.8071846043038, 18076.575401063343, 61.34888444345095, 61.46152141689355, 5.8169904927769505, 5.8718243542231034, 2445276.4592079865, 5.903217843072723], "left_speed": -100, "right_speed": -100},
      {"timestamp": "2025-06-10T09:34:42.313026", "irs": [917.5720409226201, 17926.862078492166, 60.26354599184405, 60.3653708884936, 5.824052678927418, 5.876568516467113, 2039306.3401785097, 5.908400161373787], "left_speed": -100, "right_speed": -100},
      {"timestamp": "2025-06-10T09:34:43.014123", "irs": [914.0449612562345, 17857.59777777464, 60.24825249615268, 60.34093123767951, 5.843558698245324, 5.889377836202441, 1818734.2357116633, 5.923610822812264], "left_speed": -100, "right_speed": -100},
      {"timestamp": "2025-06-10T09:34:43.616642", "irs": [926.1873493867598, 17964.2133629679, 61.96081000222681, 62.08521062807617, 5.992569432499119, 5.99414582886412, 2605753.2390339645, 6.0295360974196575], "left_speed": -100, "right_speed": -100},
      {"timestamp": "2025-06-10T09:34:44.283634", "irs": [931.9198988372259, 17919.75944896663, 61.77223415173924, 61.86312239590404, 6.183709955186755, 6.129498777061758, 3006528.637853654, 6.162442264202967], "left_speed": -100, "right_speed": -100},
      {"timestamp": "2025-06-10T09:34:44.886159", "irs": [907.7100670870909, 15708.96453164668, 189.3936863105368, 190.28023998039475, 45.65099719512156, 27.317620058594645, 872502.4789006584, 27.068207635427427], "left_speed": -100, "right_speed": -100},
      {"timestamp": "2025-06-10T09:34:45.435253", "irs": [906.9501313618806, 15685.613066436581, 190.26161287655043, 191.3659630016619, 45.881084269500505, 27.41097478058815, 856113.6010165276, 27.17287169783274], "left_speed": -100, "right_speed": -100},
      {"timestamp": "2025-06-10T09:34:46.033888", "irs": [915.6274988420577, 18157.013321711933, 61.28831300263407, 61.37119850791125, 5.799340942907201, 5.860399353797856, 2168198.683501699, 8.047754645362923], "left_speed": -100, "right_speed": -100},
      {"timestamp": "2025-06-10T09:34:46.733086", "irs": [919.721892492141, 17879.661580891432, 60.1604082333253, 60.25920808039441, 5.825519180206597, 5.8773846868888215, 2101258.0742434333, 8.327845037801945], "left_speed": -100, "right_speed": -100},
      {"timestamp": "2025-06-10T09:34:47.334438", "irs": [934.7975359936245, 17797.334960551776, 62.07945411038788, 62.18339301866183, 5.9759004386046355, 5.983980881079276, 3026893.816466441, 10.791978483586753], "left_speed": -100, "right_speed": -100},
      {"timestamp": "2025-06-10T09:34:47.933206", "irs": [925.469836988261, 17372.580347264313, 60.3047697912203, 60.39749698900664, 5.833590498550916, 5.882645070157126, 1889417.3534080435, 11.323945325567223], "left_speed": -100, "right_speed": -100},
      {"timestamp": "2025-06-10T09:34:48.533197", "irs": [915.2792097180859, 17703.058426185984, 60.31273488196317, 60.45047224159247, 5.594884669430183, 5.716837962034573, 1736296.4635379498, 11.860296214725334], "left_speed": -100, "right_speed": -100},
      {"timestamp": "2025-06-10T09:34:49.183222", "irs": [929.8351928982426, 18070.751855863487, 64.0442140033008, 64.14002021149915, 6.475926119313379, 6.334549452645669, 3112777.0140171726, 15.415613043589167], "left_speed": -100, "right_speed": -100},
      {"timestamp": "2025-06-10T09:34:49.833264", "irs": [917.9801806253155, 17913.79212281962, 60.3075218428891, 60.407871589466566, 5.8342882409253365, 5.883609530099354, 2062361.1546563453, 16.47390789914186], "left_speed": -100, "right_speed": -100}
    ]
  };

  useEffect(() => {
    // Process the data for visualization
    const processedData = jsonData.steps.map((step, index) => ({
      step: index,
      time: new Date(step.timestamp).getTime(),
      relativeTime: index * 0.5, // approximate seconds
      sensor0: step.irs[0], // BackL
      sensor1: step.irs[1], // BackR
      sensor2: step.irs[2], // FrontL
      sensor3: step.irs[3], // FrontR
      sensor4: step.irs[4], // FrontC
      sensor5: step.irs[5], // FrontRR
      sensor6: step.irs[6], // BackC
      sensor7: step.irs[7], // FrontLL
      leftSpeed: step.left_speed,
      rightSpeed: step.right_speed,
      avgFrontSensors: (step.irs[2] + step.irs[3] + step.irs[4] + step.irs[5] + step.irs[7]) / 5,
      avgBackSensors: (step.irs[0] + step.irs[1] + step.irs[6]) / 3,
      maxSensor: Math.max(...step.irs),
      minSensor: Math.min(...step.irs),
      sensorVariance: Math.sqrt(step.irs.reduce((sum, val) => sum + Math.pow(val - step.irs.reduce((a,b) => a+b)/8, 2), 0) / 8)
    }));
    setData(processedData);
  }, []);

  const sensorNames = ['BackL', 'BackR', 'FrontL', 'FrontR', 'FrontC', 'FrontRR', 'BackC', 'FrontLL'];

  // Genotype visualization data
  const genotypeData = jsonData.genotype.map((weight, index) => ({
    sensor: sensorNames[Math.floor(index / 2)],
    motor: index % 2 === 0 ? 'Left' : 'Right',
    weight: weight,
    index: index
  }));

  // Radar chart data for sensor readings at different time points
  const getRadarData = (stepIndex) => {
    if (!data[stepIndex]) return [];
    return sensorNames.map((name, i) => ({
      sensor: name,
      value: Math.log10(data[stepIndex][`sensor${i}`] + 1) * 20, // Log scale for better visualization
      rawValue: data[stepIndex][`sensor${i}`]
    }));
  };

  const obstacleDetectionData = data.map(d => ({
    step: d.step,
    frontObstacle: d.avgFrontSensors > 100 ? 1 : 0,
    backObstacle: d.avgBackSensors > 100 ? 1 : 0,
    criticalSensor: d.maxSensor > 10000 ? 1 : 0,
    variance: d.sensorVariance / 1000 // Scale down for visibility
  }));

  return (
    <div className="p-6 bg-gradient-to-br from-slate-900 to-slate-800 min-h-screen text-white">
      <div className="max-w-7xl mx-auto space-y-8">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
            🤖 Robobo Navigation Analysis
          </h1>
          <p className="text-gray-300 mt-2">Analyzing sensor data and movement patterns from 30-step robot navigation</p>
        </div>

        {/* Neural Network Genotype Visualization */}
        <div className="bg-slate-800/50 backdrop-blur rounded-xl p-6 border border-slate-700">
          <h2 className="text-2xl font-semibold mb-4 text-blue-300">🧠 Neural Network Genotype</h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={genotypeData}>
              <CartesianGrid strokeDasharray="3,3" stroke="#374151" />
              <XAxis dataKey="index" stroke="#9CA3AF" />
              <YAxis stroke="#9CA3AF" />
              <Tooltip
                contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '8px' }}
                labelStyle={{ color: '#F3F4F6' }}
              />
              <Bar dataKey="weight" fill="#8B5CF6" radius={[4, 4, 0, 0]}>
                {genotypeData.map((entry, index) => (
                  <Bar key={`cell-${index}`} fill={entry.weight > 0 ? '#10B981' : '#EF4444'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <p className="text-sm text-gray-400 mt-2">
            Green = Positive weights (forward influence), Red = Negative weights (avoidance behavior)
          </p>
        </div>

        {/* All Sensor Readings Over Time */}
        <div className="bg-slate-800/50 backdrop-blur rounded-xl p-6 border border-slate-700">
          <h2 className="text-2xl font-semibold mb-4 text-green-300">📊 All Sensor Readings Timeline</h2>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3,3" stroke="#374151" />
              <XAxis dataKey="step" stroke="#9CA3AF" />
              <YAxis scale="log" domain={[1, 'dataMax']} stroke="#9CA3AF" />
              <Tooltip
                contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '8px' }}
                labelStyle={{ color: '#F3F4F6' }}
              />
              <Legend />
              <Line type="monotone" dataKey="sensor0" stroke="#FF6B6B" strokeWidth={2} name="BackL" />
              <Line type="monotone" dataKey="sensor1" stroke="#4ECDC4" strokeWidth={2} name="BackR" />
              <Line type="monotone" dataKey="sensor2" stroke="#45B7D1" strokeWidth={2} name="FrontL" />
              <Line type="monotone" dataKey="sensor3" stroke="#FFA07A" strokeWidth={2} name="FrontR" />
              <Line type="monotone" dataKey="sensor4" stroke="#98D8C8" strokeWidth={2} name="FrontC" />
              <Line type="monotone" dataKey="sensor5" stroke="#F7DC6F" strokeWidth={2} name="FrontRR" />
              <Line type="monotone" dataKey="sensor6" stroke="#BB8FCE" strokeWidth={2} name="BackC" />
              <Line type="monotone" dataKey="sensor7" stroke="#82E0AA" strokeWidth={2} name="FrontLL" />
            </LineChart>
          </ResponsiveContainer>
          <p className="text-sm text-gray-400 mt-2">
            Log scale used due to extreme sensor value ranges. Notice the massive spikes in BackC sensor around steps 3-7!
          </p>
        </div>

        {/* Obstacle Detection Pattern */}
        <div className="bg-slate-800/50 backdrop-blur rounded-xl p-6 border border-slate-700">
          <h2 className="text-2xl font-semibold mb-4 text-red-300">🚨 Obstacle Detection Events</h2>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={obstacleDetectionData}>
              <CartesianGrid strokeDasharray="3,3" stroke="#374151" />
              <XAxis dataKey="step" stroke="#9CA3AF" />
              <YAxis stroke="#9CA3AF" />
              <Tooltip
                contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '8px' }}
                labelStyle={{ color: '#F3F4F6' }}
              />
              <Area type="monotone" dataKey="criticalSensor" stackId="1" stroke="#DC2626" fill="#DC2626" fillOpacity={0.8} name="Critical Sensor Alert" />
              <Area type="monotone" dataKey="frontObstacle" stackId="1" stroke="#F59E0B" fill="#F59E0B" fillOpacity={0.6} name="Front Obstacle" />
              <Area type="monotone" dataKey="backObstacle" stackId="1" stroke="#8B5CF6" fill="#8B5CF6" fillOpacity={0.4} name="Back Obstacle" />
            </AreaChart>
          </ResponsiveContainer>
          <p className="text-sm text-gray-400 mt-2">
            Shows when the robot detected obstacles. Critical sensor alerts indicate extremely high readings (>10k).
          </p>
        </div>

        {/* Sensor Radar Visualization */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-slate-800/50 backdrop-blur rounded-xl p-6 border border-slate-700">
            <h2 className="text-2xl font-semibold mb-4 text-purple-300">🎯 Sensor Pattern - Step 3</h2>
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart data={getRadarData(3)}>
                <PolarGrid stroke="#374151" />
                <PolarAngleAxis dataKey="sensor" tick={{ fill: '#9CA3AF', fontSize: 12 }} />
                <PolarRadiusAxis stroke="#9CA3AF" />
                <Radar name="Sensor Reading" dataKey="value" stroke="#8B5CF6" fill="#8B5CF6" fillOpacity={0.3} strokeWidth={2} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '8px' }}
                  labelStyle={{ color: '#F3F4F6' }}
                  formatter={(value, name, props) => [props.payload.rawValue.toFixed(1), 'Raw Value']}
                />
              </RadarChart>
            </ResponsiveContainer>
            <p className="text-sm text-gray-400 mt-2">Major obstacle detected in BackC sensor!</p>
          </div>

          <div className="bg-slate-800/50 backdrop-blur rounded-xl p-6 border border-slate-700">
            <h2 className="text-2xl font-semibent mb-4 text-purple-300">🎯 Sensor Pattern - Step 8</h2>
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart data={getRadarData(8)}>
                <PolarGrid stroke="#374151" />
                <PolarAngleAxis dataKey="sensor" tick={{ fill: '#9CA3AF', fontSize: 12 }} />
                <PolarRadiusAxis stroke="#9CA3AF" />
                <Radar name="Sensor Reading" dataKey="value" stroke="#10B981" fill="#10B981" fillOpacity={0.3} strokeWidth={2} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '8px' }}
                  labelStyle={{ color: '#F3F4F6' }}
                  formatter={(value, name, props) => [props.payload.rawValue.toFixed(1), 'Raw Value']}
                />
              </RadarChart>
            </ResponsiveContainer>
            <p className="text-sm text-gray-400 mt-2">Front sensors showing obstacle pattern</p>
          </div>
        </div>

        {/* Front vs Back Sensor Comparison */}
        <div className="bg-slate-800/50 backdrop-blur rounded-xl p-6 border border-slate-700">
          <h2 className="text-2xl font-semibold mb-4 text-yellow-300">⚖️ Front vs Back Sensor Averages</h2>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={data}>
              <CartesianGrid strokeDasharray="3,3" stroke="#374151" />
              <XAxis dataKey="step" stroke="#9CA3AF" />
              <YAxis scale="log" domain={[1, 'dataMax']} stroke="#9CA3AF" />
              <Tooltip
                contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '8px' }}
                labelStyle={{ color: '#F3F4F6' }}
              />
              <Area type="monotone" dataKey="avgBackSensors" stroke="#F59E0B" fill="#F59E0B" fillOpacity={0.6} name="Back Sensors Avg" />
              <Area type="monotone" dataKey="avgFrontSensors" stroke="#3B82F6" fill="#3B82F6" fillOpacity={0.4} name="Front Sensors Avg" />
            </AreaChart>
          </ResponsiveContainer>
          <p className="text-sm text-gray-400 mt-2">
            Robot seems to be backing into obstacles (high back sensor readings) while front sensors remain relatively stable.
          </p>
        </div>

        {/* Sensor Variance Analysis */}
        <div className="bg-slate-800/50 backdrop-blur rounded-xl p-6 border border-slate-700">
          <h2 className="text-2xl font-semibold mb-4 text-indigo-300">📈 Sensor Reading Variance</h2>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3,3" stroke="#374151" />
              <XAxis dataKey="step" stroke="#9CA3AF" />
              <YAxis stroke="#9CA3AF" />
              <Tooltip
                contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '8px' }}
                labelStyle={{ color: '#F3F4F6' }}
              />
              <Line type="monotone" dataKey="sensorVariance" stroke="#6366F1" strokeWidth={3} name="Sensor Variance" />
            </LineChart>
          </ResponsiveContainer>
          <p className="text-sm text-gray-400 mt-2">
            High variance indicates uneven sensor readings - robot is encountering complex obstacle patterns.
          </p>
        </div>

        {/* Key Insights */}
        <div className="bg-gradient-to-r from-slate-800 to-slate-700 rounded-xl p-6 border-l-4 border-blue-500">
          <h2 className="text-2xl font-semibold mb-4 text-blue-300">🔍 Key Insights</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
            <div className="space-y-2">
              <p><span className="text-green-400">✓</span> Robot consistently moves backward (both motors at -100)</p>
              <p><span className="text-yellow-400">⚠</span> BackC sensor shows extreme readings (up to 3M+ units)</p>
              <p><span className="text-red-400">⚠</span> Major obstacle encounters at steps 3-7</p>
            </div>
            <div className="space-y-2">
              <p><span className="text-blue-400">ℹ</span> Front sensors remain relatively stable (60-200 range)</p>
              <p><span className="text-purple-400">ℹ</span> High sensor variance indicates complex environment</p>
              <p><span className="text-green-400">✓</span> Neural network has strong avoidance behavior (negative weights)</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// Log to console
console.log('Hello console')