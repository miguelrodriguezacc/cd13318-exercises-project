#!/usr/bin/env python3
"""
Script de diagnóstico para inspeccionar ChromaDB
Útil para entender qué metadatos se están guardando
"""
 
import chromadb
from pathlib import Path
import json
 
def diagnose_chroma():
    """
    Inspecciona la estructura de ChromaDB y los metadatos
    """
    
    print("=" * 70)
    print("🔍 DIAGNÓSTICO DE CHROMADB")
    print("=" * 70)
    
    # Paso 1: Busca directorios de ChromaDB
    print("\n📁 PASO 1: Buscando directorios de ChromaDB...")
    
    current_dir = Path(".")
    chroma_dirs = [d for d in current_dir.iterdir() 
                   if d.is_dir() and 'chroma' in d.name.lower()]
    
    if not chroma_dirs:
        print("   ❌ No se encontraron directorios de ChromaDB")
        print("   Busca carpetas con 'chroma' en su nombre")
        return False
    
    print(f"   ✅ Encontrados {len(chroma_dirs)} directorio(s) de ChromaDB:")
    for d in chroma_dirs:
        print(f"      - {d.name}")
    
    # Paso 2: Inspecciona cada directorio
    for chroma_dir in chroma_dirs:
        print(f"\n{'='*70}")
        print(f"📊 INSPECCIONANDO: {chroma_dir.name}")
        print(f"{'='*70}")
        
        try:
            client = chromadb.PersistentClient(path=str(chroma_dir))
            collections = client.list_collections()
            
            if not collections:
                print("   ❌ Sin colecciones en este directorio")
                continue
            
            print(f"   ✅ Colecciones encontradas: {len(collections)}")
            
            for collection in collections:
                print(f"\n   📦 Colección: {collection.name}")
                
                # Obtén estadísticas
                try:
                    count = collection.count()
                    print(f"      Total documentos: {count}")
                except:
                    count = "N/A"
                    print(f"      Total documentos: N/A")
                
                # Obtén metadatos de ejemplo
                data = collection.get(limit=5)
                
                if not data.get("metadatas"):
                    print(f"      ❌ SIN METADATOS")
                    print(f"         (Este es el problema)")
                    continue
                
                # Paso 3: Inspecciona los campos de metadatos
                print(f"\n      📋 Campos de metadatos disponibles:")
                
                # Obtén todos los campos únicos
                all_fields = set()
                for meta in data["metadatas"]:
                    all_fields.update(meta.keys())
                
                for field in sorted(all_fields):
                    print(f"         ✅ {field}")
                
                # Paso 4: Verifica si existe 'mission'
                print(f"\n      🎯 VERIFICACIÓN DE CAMPO 'mission':")
                
                if "mission" in all_fields:
                    print(f"         ✅ Campo 'mission' EXISTE")
                    
                    # Muestra valores únicos de mission
                    missions = set()
                    for meta in data["metadatas"]:
                        if "mission" in meta:
                            missions.add(meta["mission"])
                    
                    print(f"         Valores encontrados: {sorted(missions)}")
                    
                    # Verifica que no haya "unknown"
                    if "unknown" in missions or len(missions) == 0:
                        print(f"         ⚠️  ADVERTENCIA: Solo hay 'unknown' o vacío")
                        print(f"            Necesitas arreglarlo")
                else:
                    print(f"         ❌ Campo 'mission' NO EXISTE")
                    print(f"         ⚠️  ESTE ES EL PROBLEMA")
                
                # Paso 5: Muestra ejemplos de metadatos
                print(f"\n      📄 Ejemplos de metadatos:")
                for idx, meta in enumerate(data["metadatas"][:3], 1):
                    print(f"\n         Ejemplo {idx}:")
                    for key, value in meta.items():
                        if isinstance(value, str) and len(value) > 50:
                            print(f"            {key}: {value[:50]}...")
                        else:
                            print(f"            {key}: {value}")
                
                # Paso 6: Diagnóstico
                print(f"\n      🔧 DIAGNÓSTICO:")
                
                if "mission" not in all_fields:
                    print(f"         ❌ PROBLEMA: Falta el campo 'mission'")
                    print(f"")
                    print(f"         SOLUCIÓN 1 (Rápida):")
                    print(f"         Ejecuta:")
                    print(f"         $ python fix_mission_metadata.py")
                    print(f"")
                    print(f"         SOLUCIÓN 2 (Completa):")
                    print(f"         Ejecuta:")
                    print(f"         $ python embedding_pipeline_mejorado.py")
                else:
                    missions = {m.get("mission") for m in data["metadatas"]}
                    
                    if "unknown" in missions or len(missions) == 1:
                        print(f"         ⚠️  PROBLEMA: Misiones no identificadas correctamente")
                        print(f"            Valores: {missions}")
                        print(f"")
                        print(f"         SOLUCIÓN:")
                        print(f"         Ejecuta:")
                        print(f"         $ python embedding_pipeline_mejorado.py")
                    else:
                        print(f"         ✅ TODO PARECE ESTAR CORRECTO")
                        print(f"            Misiones encontradas: {missions}")
                        print(f"            Intenta limpiar caché:")
                        print(f"            $ streamlit cache clear")
                        print(f"            $ streamlit run chat.py")
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}\n")
    return True
 
if __name__ == "__main__":
    diagnose_chroma()
 