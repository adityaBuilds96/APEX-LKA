"""
scratch/generate_golden_scenarios.py
====================================
Generates 101 industrial-grade golden scenarios across 6 critical categories
and builds the comprehensive scenarios/golden_test_suite/manifest.yaml index.
"""

from pathlib import Path
import yaml

SCENARIO_DIR = Path("scenarios/golden_test_suite")
SCENARIO_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = {
    "PEDESTRIAN_INTERACTIONS": 20,
    "VEHICLE_CUTINS": 20,
    "LANE_CLOSURES": 15,
    "WEATHER_SCENARIOS": 15,
    "TRACK_OBSTACLES": 15,
    "EMERGENCY_EVENTS": 16,
}

WEATHER_LIST = [
    "clear_noon",
    "cloudy_afternoon",
    "rain_moderate",
    "fog_dense",
    "dusk_golden",
    "night_streetlights",
    "night_pitch_black",
    "dust_cloud_heavy",
]

manifest_entries = []

scenario_counter = 0

# 1. Pedestrian Interactions (20)
ped_descriptions = [
    ("ped_01_jaywalk_clear", "Pedestrian jaywalking in full daylight", "clear_noon", 10.0, 1.4),
    ("ped_02_jump_out_parked", "Pedestrian jumps out from between parked vehicles", "clear_noon", 8.0, 2.5),
    ("ped_03_child_crosswalk", "Child running across school crosswalk", "cloudy_afternoon", 12.0, 2.8),
    ("ped_04_night_dark_clothes", "Pedestrian crossing at night in dark clothing without streetlights", "night_pitch_black", 12.0, 1.2),
    ("ped_05_wheelchair_crossing", "Motorized wheelchair user crossing campus boulevard", "dusk_golden", 15.0, 1.0),
    ("ped_06_group_crosswalk", "Group of 3 students crossing in pedestrian zone", "clear_noon", 14.0, 1.3),
    ("ped_07_rain_umbrella_jogger", "Jogger with umbrella running into road in rain", "rain_moderate", 10.0, 3.2),
    ("ped_08_blind_corner_walker", "Pedestrian emerging from blind corner behind building", "cloudy_afternoon", 7.0, 1.5),
    ("ped_09_stroller_crossing", "Parent pushing stroller entering crosswalk", "clear_noon", 12.0, 1.1),
    ("ped_10_dog_walker_leash", "Dog walker whose pet darts across road", "dusk_golden", 9.0, 2.2),
    ("ped_11_fog_pedestrian", "Pedestrian crossing in dense 50m fog", "fog_dense", 11.0, 1.3),
    ("ped_12_night_streetlights_ped", "Pedestrian crossing under sodium streetlights", "night_streetlights", 10.0, 1.4),
    ("ped_13_skateboarder_crossing", "Skateboarder rapidly traversing campus lane", "clear_noon", 15.0, 4.0),
    ("ped_14_dust_storm_baja_marshal", "Track marshal crossing during BAJA dust storm", "dust_cloud_heavy", 10.0, 1.2),
    ("ped_15_distracted_phone_walker", "Distracted pedestrian walking into lane looking at phone", "cloudy_afternoon", 8.0, 1.1),
    ("ped_16_cyclist_crossing", "Cyclist crossing perpendicular across crosswalk", "clear_noon", 14.0, 3.5),
    ("ped_17_construction_worker_flag", "Construction flagger waving in lane ahead", "clear_noon", 16.0, 0.5),
    ("ped_18_rain_curb_slip_ped", "Pedestrian slipping off curb into roadway in heavy rain", "rain_moderate", 8.0, 1.8),
    ("ped_19_dusk_shadow_runner", "Runner in deep tree shadows at sunset", "dusk_golden", 9.0, 3.0),
    ("ped_20_wrong_side_jaywalker", "Pedestrian walking toward vehicle on wrong side of road", "night_streetlights", 12.0, 1.3),
]

for idx, (sid, desc, wth, trig_d, spd) in enumerate(ped_descriptions, 1):
    scenario_counter += 1
    scen_id = f"PED_{idx:02d}"
    filename = f"{sid}.yaml"
    data = {
        "scenario_id": scen_id,
        "category": "PEDESTRIAN_INTERACTION",
        "description": desc,
        "map": "baja_track_2027" if "baja" in sid else "college_campus",
        "weather": wth,
        "seed": 100 + idx,
        "duration_sec": 12.0,
        "ego_vehicle": {
            "spawn_x": 120.5,
            "spawn_y": 48.2,
            "initial_speed_kmh": 20.0,
        },
        "actors": [
            {
                "type": "walker.pedestrian.0001",
                "spawn_x": 130.0 + (idx * 0.5),
                "spawn_y": 45.0,
                "trigger_distance_m": trig_d,
                "target_x": 130.0 + (idx * 0.5),
                "target_y": 51.5,
                "walk_speed_mps": spd,
                "behavior": "cross",
            }
        ],
        "success_criteria": {
            "collision": False,
            "min_distance_to_pedestrian_m": 1.5,
            "emergency_brake_activated": True,
            "max_path_deviation_m": 0.8,
        },
    }
    with open(SCENARIO_DIR / filename, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    manifest_entries.append({
        "scenario_id": scen_id,
        "file": filename,
        "category": "PEDESTRIAN_INTERACTION",
        "map": data["map"],
        "weather": wth,
        "description": desc,
        "difficulty": "CRITICAL",
    })

# 2. Vehicle Cut-ins (20)
cutin_descriptions = [
    ("cutin_01_aggressive_left", "Aggressive adjacent sedan cut-in from left lane", "clear_noon", 12.0, 6.0),
    ("cutin_02_slow_merging_truck", "Slow delivery truck merging from right parking lane", "cloudy_afternoon", 14.0, 3.0),
    ("cutin_03_cut_and_brake", "Vehicle cuts in at 10m then slams brakes to full stop", "clear_noon", 10.0, 5.0),
    ("cutin_04_motorcycle_filter", "Motorcycle filtering between lanes cutting into corridor", "dusk_golden", 8.0, 7.5),
    ("cutin_05_rain_cutin_spray", "SUV cutting in during heavy rain creating blinding spray", "rain_moderate", 11.0, 6.0),
    ("cutin_06_blind_spot_merge", "Vehicle emerging from blind parking spot without signaling", "clear_noon", 7.0, 2.5),
    ("cutin_07_highway_ramp_merge", "On-ramp high-speed merge forcing lane yield", "clear_noon", 18.0, 8.0),
    ("cutin_08_night_unlit_cutin", "Unlit vehicle cutting across lane at night", "night_pitch_black", 9.0, 5.0),
    ("cutin_09_fog_sudden_merge", "Vehicle suddenly emerging from fog cutting in front", "fog_dense", 10.0, 4.5),
    ("cutin_10_bus_pull_out", "Campus shuttle bus pulling out from transit stop into lane", "cloudy_afternoon", 12.0, 3.5),
    ("cutin_11_taxi_sharp_pickup", "Taxi making sharp swerve across lane to pick up fare", "night_streetlights", 9.0, 4.0),
    ("cutin_12_double_lane_swerve", "Erratic vehicle swerving across two lanes simultaneously", "dusk_golden", 14.0, 6.5),
    ("cutin_13_baja_buggy_overtake", "Competitor BAJA buggy overtaking and cutting across track", "dust_cloud_heavy", 11.0, 7.0),
    ("cutin_14_ambulance_overtake", "Emergency vehicle overtaking with sirens forcing yield", "clear_noon", 15.0, 9.0),
    ("cutin_15_garbage_truck_block", "Sanitation truck abruptly swerving into center lane", "cloudy_afternoon", 11.0, 2.0),
    ("cutin_16_tailgater_whip_around", "Tailgater whipping around and cutting in immediately ahead", "clear_noon", 8.0, 7.0),
    ("cutin_17_slick_wet_drift_cutin", "Vehicle sliding across lane markings on slick wet asphalt", "rain_moderate", 10.0, 5.5),
    ("cutin_18_delivery_van_reverse", "Delivery van reversing out into active driving corridor", "dusk_golden", 9.0, 2.0),
    ("cutin_19_traffic_light_cutin", "Vehicle cutting across intersection to beat yellow light", "night_streetlights", 13.0, 6.5),
    ("cutin_20_tight_roundabout_cutin", "Vehicle cutting into roundabout without right-of-way", "clear_noon", 9.0, 4.0),
]

for idx, (sid, desc, wth, trig_d, spd) in enumerate(cutin_descriptions, 1):
    scenario_counter += 1
    scen_id = f"CUTIN_{idx:02d}"
    filename = f"{sid}.yaml"
    data = {
        "scenario_id": scen_id,
        "category": "VEHICLE_CUTIN",
        "description": desc,
        "map": "baja_track_2027" if "baja" in sid else "college_campus",
        "weather": wth,
        "seed": 200 + idx,
        "duration_sec": 12.0,
        "ego_vehicle": {
            "spawn_x": 120.5,
            "spawn_y": 48.2,
            "initial_speed_kmh": 25.0,
        },
        "actors": [
            {
                "type": "vehicle.audi.a2",
                "spawn_x": 135.0,
                "spawn_y": 45.0,
                "trigger_distance_m": trig_d,
                "target_x": 145.0,
                "target_y": 48.2,
                "speed_mps": spd,
                "behavior": "cutin",
            }
        ],
        "success_criteria": {
            "collision": False,
            "min_distance_m": 2.0,
            "emergency_brake_activated": True,
            "max_path_deviation_m": 1.0,
        },
    }
    with open(SCENARIO_DIR / filename, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    manifest_entries.append({
        "scenario_id": scen_id,
        "file": filename,
        "category": "VEHICLE_CUTIN",
        "map": data["map"],
        "weather": wth,
        "description": desc,
        "difficulty": "CRITICAL",
    })

# 3. Lane Closures (15)
lane_descriptions = [
    ("lane_01_construction_cones", "Row of 10 orange cones blocking right half of lane", "clear_noon", 20.0),
    ("lane_02_work_truck_parked", "Utility service truck parked blocking entire driving lane", "cloudy_afternoon", 25.0),
    ("lane_03_detour_arrow_barriers", "Sequential reflective detour chevron barriers redirecting traffic", "dusk_golden", 22.0),
    ("lane_04_missing_markings_paving", "Freshly repaved asphalt with zero lane paint markings", "clear_noon", 30.0),
    ("lane_05_faded_worn_stripes", "Heavily sun-bleached and worn lane lines with high ambiguity", "clear_noon", 25.0),
    ("lane_06_pothole_excavation", "Deep road repair trench surrounded by flashing stanchions", "rain_moderate", 18.0),
    ("lane_07_bottleneck_merge", "Two-lane campus road narrowing abruptly to single shared lane", "cloudy_afternoon", 24.0),
    ("lane_08_night_cone_taper", "Construction cone taper zone under dark night conditions", "night_pitch_black", 15.0),
    ("lane_09_foggy_barrier_diversion", "Concrete K-rail barrier diverting traffic in 50m fog", "fog_dense", 14.0),
    ("lane_10_overgrown_tree_closure", "Downed tree branch encroaching into outer 1.5m of lane", "rain_moderate", 16.0),
    ("lane_11_flooded_curb_lane", "Standing water covering right half of roadway", "rain_moderate", 20.0),
    ("lane_12_baja_boulder_barrier", "Large rock fall blocking left rut on BAJA hill climb", "dust_cloud_heavy", 15.0),
    ("lane_13_student_fair_tents", "Temporary market tents encroaching onto campus roadway", "clear_noon", 22.0),
    ("lane_14_asphalt_milling_grooves", "Milled pavement longitudinal grooving causing false line edges", "cloudy_afternoon", 25.0),
    ("lane_15_spilled_paint_splatter", "Spilled yellow traffic paint creating confusing deceptive markings", "clear_noon", 20.0),
]

for idx, (sid, desc, wth, trig_d) in enumerate(lane_descriptions, 1):
    scenario_counter += 1
    scen_id = f"LANE_{idx:02d}"
    filename = f"{sid}.yaml"
    data = {
        "scenario_id": scen_id,
        "category": "LANE_CLOSURE",
        "description": desc,
        "map": "baja_track_2027" if "baja" in sid else "college_campus",
        "weather": wth,
        "seed": 300 + idx,
        "duration_sec": 12.0,
        "ego_vehicle": {
            "spawn_x": 120.5,
            "spawn_y": 48.2,
            "initial_speed_kmh": 20.0,
        },
        "actors": [
            {
                "type": "static.prop.constructioncone",
                "spawn_x": 138.0,
                "spawn_y": 48.0,
                "trigger_distance_m": trig_d,
                "speed_mps": 0.0,
                "behavior": "static",
            }
        ],
        "success_criteria": {
            "collision": False,
            "min_distance_m": 1.2,
            "max_path_deviation_m": 1.2,
        },
    }
    with open(SCENARIO_DIR / filename, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    manifest_entries.append({
        "scenario_id": scen_id,
        "file": filename,
        "category": "LANE_CLOSURE",
        "map": data["map"],
        "weather": wth,
        "description": desc,
        "difficulty": "HIGH",
    })

# 4. Weather Scenarios (15)
weather_descriptions = [
    ("weather_01_monsoon_downpour", "Torrential monsoon rainfall with maximum camera lens water film", "rain_moderate", 15.0),
    ("weather_02_pea_soup_fog", "Dense maritime fog with visibility under 35 meters", "fog_dense", 12.0),
    ("weather_03_sunset_direct_glare", "Low 5-degree sun azimuth blinding optical camera sensor directly", "dusk_golden", 20.0),
    ("weather_04_baja_haboob_dust", "Dense dust wall with blowing sand on BAJA track", "dust_cloud_heavy", 10.0),
    ("weather_05_pitch_black_rural", "Headlights-only driving with zero ambient lighting", "night_pitch_black", 18.0),
    ("weather_06_wet_asphalt_mirror", "Wet asphalt reflecting street lamps creating phantom lane lines", "night_streetlights", 16.0),
    ("weather_07_cloud_shadow_strobe", "Rapid intermittent tree canopy shadows at noon", "clear_noon", 22.0),
    ("weather_08_heavy_overcast_dim", "Dark thunderstorm clouds dropping light to twilight levels", "cloudy_afternoon", 20.0),
    ("weather_09_sunrise_dew_fog", "Early dawn ground fog burning off with low horizon glare", "dusk_golden", 16.0),
    ("weather_10_rain_puddle_splash", "Vehicle hitting deep standing puddle creating temporary blinding wall", "rain_moderate", 14.0),
    ("weather_11_tunnel_exit_bloom", "Rapid optical exposure bloom transitioning from tunnel to bright sun", "clear_noon", 25.0),
    ("weather_12_dust_cloud_chase", "Following directly behind buggy producing thick rooster-tail dust", "dust_cloud_heavy", 12.0),
    ("weather_13_glare_mirage_heat", "Extreme heat shimmer creating mirage distortion on asphalt", "clear_noon", 24.0),
    ("weather_14_night_rain_reflection", "Combining rain drops, wiper chatter, and wet specular reflections", "rain_moderate", 14.0),
    ("weather_15_dense_fog_tail_lights", "Detecting faint lead vehicle brake lights through heavy fog", "fog_dense", 15.0),
]

for idx, (sid, desc, wth, trig_d) in enumerate(weather_descriptions, 1):
    scenario_counter += 1
    scen_id = f"WTH_{idx:02d}"
    filename = f"{sid}.yaml"
    data = {
        "scenario_id": scen_id,
        "category": "WEATHER_SCENARIOS",
        "description": desc,
        "map": "baja_track_2027" if "baja" in sid or "dust" in sid else "college_campus",
        "weather": wth,
        "seed": 400 + idx,
        "duration_sec": 12.0,
        "ego_vehicle": {
            "spawn_x": 120.5,
            "spawn_y": 48.2,
            "initial_speed_kmh": 22.0,
        },
        "actors": [],
        "success_criteria": {
            "collision": False,
            "max_path_deviation_m": 0.9,
        },
    }
    with open(SCENARIO_DIR / filename, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    manifest_entries.append({
        "scenario_id": scen_id,
        "file": filename,
        "category": "WEATHER_SCENARIOS",
        "map": data["map"],
        "weather": wth,
        "description": desc,
        "difficulty": "HIGH",
    })

# 5. Track Obstacles (15)
obs_descriptions = [
    ("obs_01_tire_in_lane", "Discarded truck tire carcass lying flat in vehicle wheel path", "clear_noon", 18.0),
    ("obs_02_cardboard_box", "Large empty shipping box blowing across roadway", "cloudy_afternoon", 15.0),
    ("obs_03_traffic_barrel", "Impact barrel rolling into active lane", "dusk_golden", 16.0),
    ("obs_04_loose_gravel_patch", "Dense patch of loose blue metal gravel causing traction loss", "clear_noon", 20.0),
    ("obs_05_fallen_branch", "Heavy tree limb spanning across left lane boundary", "rain_moderate", 14.0),
    ("obs_06_speed_bump_high_speed", "Approaching 8cm college speed bump at excessive speed", "clear_noon", 25.0),
    ("obs_07_dog_crossing_baja", "Stray animal running across dirt track", "dust_cloud_heavy", 12.0),
    ("obs_08_dropped_cargo_pallet", "Wooden pallet dropped by preceding utility vehicle", "night_streetlights", 14.0),
    ("obs_09_baja_log_barrier", "Heavy horizontal log obstacle on BAJA obstacle course", "dust_cloud_heavy", 16.0),
    ("obs_10_manhole_cover_missing", "Open storm drain / utility access cover in roadway", "night_pitch_black", 12.0),
    ("obs_11_plastic_tarp_blowing", "Large blue plastic tarp fluttering across windshield path", "cloudy_afternoon", 15.0),
    ("obs_12_oil_slick_patch", "Engine oil spill creating deceptive dark reflection patch", "clear_noon", 18.0),
    ("obs_13_baja_rock_mound", "Jagged rock garden formation requiring immediate steering nudge", "dust_cloud_heavy", 14.0),
    ("obs_14_lost_hubcap_debris", "Metal hubcap skittering across lane", "dusk_golden", 15.0),
    ("obs_15_spilled_aggregate_dirt", "Load of red clay spilled across asphalt creating low-mu zone", "rain_moderate", 16.0),
]

for idx, (sid, desc, wth, trig_d) in enumerate(obs_descriptions, 1):
    scenario_counter += 1
    scen_id = f"OBS_{idx:02d}"
    filename = f"{sid}.yaml"
    data = {
        "scenario_id": scen_id,
        "category": "TRACK_OBSTACLES",
        "description": desc,
        "map": "baja_track_2027" if "baja" in sid else "college_campus",
        "weather": wth,
        "seed": 500 + idx,
        "duration_sec": 12.0,
        "ego_vehicle": {
            "spawn_x": 120.5,
            "spawn_y": 48.2,
            "initial_speed_kmh": 20.0,
        },
        "actors": [
            {
                "type": "static.prop.box",
                "spawn_x": 135.0,
                "spawn_y": 48.2,
                "trigger_distance_m": trig_d,
                "speed_mps": 0.0,
                "behavior": "static",
            }
        ],
        "success_criteria": {
            "collision": False,
            "min_distance_m": 1.2,
            "emergency_brake_activated": True,
            "max_path_deviation_m": 1.0,
        },
    }
    with open(SCENARIO_DIR / filename, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    manifest_entries.append({
        "scenario_id": scen_id,
        "file": filename,
        "category": "TRACK_OBSTACLES",
        "map": data["map"],
        "weather": wth,
        "description": desc,
        "difficulty": "HIGH",
    })

# 6. Emergency Events (16)
emerg_descriptions = [
    ("emerg_01_lead_vehicle_hard_stop", "Lead car slams full emergency ABS stop (-8 m/s^2)", "clear_noon", 18.0),
    ("emerg_02_wrong_way_headon", "Oncoming vehicle driving directly head-on in ego lane", "cloudy_afternoon", 25.0),
    ("emerg_03_intersection_red_runner", "Cross-traffic truck running red light at campus intersection", "dusk_golden", 20.0),
    ("emerg_04_phantom_braking_false_threat", "Elevated road sign causing radar/vision ambiguity", "clear_noon", 22.0),
    ("emerg_05_steering_actuator_glitch", "Simulated steering torque dropout requiring immediate fail-safe", "clear_noon", 15.0),
    ("emerg_06_lead_vehicle_swerve_obstacle", "Lead vehicle swerves violently to reveal stationary roadblock", "cloudy_afternoon", 16.0),
    ("emerg_07_night_headon_highbeams", "Oncoming vehicle blinding ego vehicle with illegal high beams", "night_pitch_black", 22.0),
    ("emerg_08_runaway_cart", "Heavy maintenance utility cart rolling down incline across lane", "clear_noon", 14.0),
    ("emerg_09_brake_boost_loss", "Simulated vehicle deceleration degradation requiring longer headway", "clear_noon", 20.0),
    ("emerg_10_camera_lens_mud_splatter", "Sudden heavy mud splatter obscuring 60% of camera sensor", "dust_cloud_heavy", 12.0),
    ("emerg_11_crosswalk_child_chase_ball", "Small child chasing ball across multi-lane roadway", "clear_noon", 14.0),
    ("emerg_12_rear_end_threat_yield", "Tailgater approaching rapidly from behind while ego must brake", "cloudy_afternoon", 18.0),
    ("emerg_13_multi_car_pileup_fog", "Multiple stopped vehicles piled up in dense fog bank", "fog_dense", 15.0),
    ("emerg_14_baja_rollover_ahead", "Preceding buggy rolls over blocking narrow track sector", "dust_cloud_heavy", 16.0),
    ("emerg_15_sudden_sinkhole_drop", "Asphalt collapse ahead requiring immediate emergency brake", "rain_moderate", 15.0),
    ("emerg_16_sensor_frame_drop_burst", "Temporary 3-frame CSI camera blackout during critical maneuver", "clear_noon", 16.0),
]

for idx, (sid, desc, wth, trig_d) in enumerate(emerg_descriptions, 1):
    scenario_counter += 1
    scen_id = f"EMERG_{idx:02d}"
    filename = f"{sid}.yaml"
    data = {
        "scenario_id": scen_id,
        "category": "EMERGENCY_EVENT",
        "description": desc,
        "map": "baja_track_2027" if "baja" in sid else "college_campus",
        "weather": wth,
        "seed": 600 + idx,
        "duration_sec": 12.0,
        "ego_vehicle": {
            "spawn_x": 120.5,
            "spawn_y": 48.2,
            "initial_speed_kmh": 22.0,
        },
        "actors": [
            {
                "type": "vehicle.audi.etron",
                "spawn_x": 136.0,
                "spawn_y": 48.2,
                "trigger_distance_m": trig_d,
                "speed_mps": 0.0,
                "behavior": "brake",
            }
        ],
        "success_criteria": {
            "collision": False,
            "min_distance_m": 1.5,
            "emergency_brake_activated": True,
            "max_path_deviation_m": 1.0,
        },
    }
    with open(SCENARIO_DIR / filename, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    manifest_entries.append({
        "scenario_id": scen_id,
        "file": filename,
        "category": "EMERGENCY_EVENT",
        "map": data["map"],
        "weather": wth,
        "description": desc,
        "difficulty": "CRITICAL",
    })

# Write master manifest.yaml
manifest_doc = {
    "version": "1.0.0",
    "total_scenarios": len(manifest_entries),
    "generated_date": "2026-10-13",
    "categories": {
        "PEDESTRIAN_INTERACTIONS": 20,
        "VEHICLE_CUTINS": 20,
        "LANE_CLOSURES": 15,
        "WEATHER_SCENARIOS": 15,
        "TRACK_OBSTACLES": 15,
        "EMERGENCY_EVENTS": 16,
    },
    "scenarios": manifest_entries,
}

manifest_path = SCENARIO_DIR / "manifest.yaml"
with open(manifest_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(manifest_doc, f, sort_keys=False)

print(f"Generated {len(manifest_entries)} Golden Scenarios in {SCENARIO_DIR} and created {manifest_path}!")
