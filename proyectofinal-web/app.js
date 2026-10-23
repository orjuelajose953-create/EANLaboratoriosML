// Motor de predicción: replica el pipeline de scikit-learn guardado en modelo.js
// (imputación + one-hot + escalado + promedio de los árboles del bosque).

// Los árboles de sklearn comparan en float32: se pasan umbrales y entradas a float32 para coincidir exacto.
for (const a of MODELO.arboles) a.t = Float32Array.from(a.t);

function vectorizar(entrada) {
  const x = [];
  for (const c of MODELO.categoricas) {
    const v = entrada[c.nombre] == null || entrada[c.nombre] === "" ? c.relleno : String(entrada[c.nombre]);
    const uno = new Array(c.categorias.length + (c.tiene_infrecuente ? 1 : 0)).fill(0);
    const k = c.categorias.indexOf(v);
    if (k >= 0) uno[k] = 1;
    else if (c.tiene_infrecuente) uno[uno.length - 1] = 1;
    x.push(...uno);
  }
  const n = MODELO.numericas;
  const crudos = n.nombres.map((nombre) => entrada[nombre]);
  const falta = crudos.map((v) => v == null || v === "" || Number.isNaN(Number(v)));
  const vals = crudos.map((v, j) => (falta[j] ? n.medianas[j] : Number(v)));
  for (const j of n.indicadores) vals.push(falta[j] ? 1 : 0);
  vals.forEach((v, j) => x.push((v - n.media[j]) / n.escala[j]));
  return Float32Array.from(x);
}

function predecir(entrada) {
  const x = vectorizar(entrada);
  let suma = 0;
  for (const a of MODELO.arboles) {
    let nodo = 0;
    while (a.f[nodo] !== -1) nodo = x[a.f[nodo]] <= a.t[nodo] ? a.i[nodo] : a.d[nodo];
    suma += a.p[nodo];
  }
  return suma / MODELO.arboles.length;
}

// En la consola del navegador: probarModelo() compara contra las probabilidades calculadas en Python.
function probarModelo() {
  if (!MODELO.casos_prueba) {
    console.info("Esta copia pública no incluye órdenes de prueba.");
    return null;
  }
  const filas = MODELO.casos_prueba.map((c) => {
    const p = predecir(c);
    return { esperada: c.p_esperada, navegador: +p.toFixed(4), ok: Math.abs(p - c.p_esperada) < 1e-3 };
  });
  console.table(filas);
  return filas.every((f) => f.ok);
}

// ---------- Formulario ----------

const ETIQUETAS = {
  mercado: "Mercado", servicio: "Servicio", centro_trabajo: "Centro de trabajo", region: "Región",
  municipio: "Municipio", clase_medidor: "Clase de medidor", uso: "Uso", firma: "Firma instaladora",
  tipo_solicitante: "Tipo de solicitante", tiene_poliza: "Tiene póliza",
  presion_contratada: "Presión contratada", n_artefactos: "Número de artefactos",
  carga_artefactos: "Carga total de artefactos", n_puntos_taponados: "Puntos taponados",
  longitud_instalacion_m: "Longitud de la instalación (m)", tiene_calentador: "Tiene calentador",
  tiene_horno: "Tiene horno", tiene_secadora_chimenea_bbq: "Tiene secadora, chimenea o BBQ",
  fecha_entrada: "Fecha de entrada", unidades_edificio: "Órdenes del mismo edificio a la fecha",
  unidades_oportunidad: "Órdenes de la misma oportunidad a la fecha",
};
const CASILLAS = ["tiene_calentador", "tiene_horno", "tiene_secadora_chimenea_bbq"];
// Variables que el usuario no escribe directamente: se calculan desde otras entradas
const DERIVADAS = ["mes_sen", "mes_cos", "dia_semana", "unidades_edificio_a_la_fecha", "unidades_oportunidad_a_la_fecha"];
const OTRO = "__otro__";

function crearCampo(nombre, control) {
  const etiqueta = document.createElement("label");
  etiqueta.textContent = ETIQUETAS[nombre] || nombre;
  control.id = control.name = nombre;
  etiqueta.htmlFor = nombre;
  const fila = document.createElement("div");
  fila.className = "campo";
  fila.append(etiqueta, control);
  return fila;
}

function construirFormulario(form) {
  for (const c of MODELO.categoricas) {
    const sel = document.createElement("select");
    for (const v of c.categorias) sel.add(new Option(v, v));
    if (c.tiene_infrecuente) sel.add(new Option("Otro / no listado", OTRO));
    form.append(crearCampo(c.nombre, sel));
  }
  const fecha = document.createElement("input");
  fecha.type = "date";
  fecha.value = new Date().toISOString().slice(0, 10);
  form.append(crearCampo("fecha_entrada", fecha));
  for (const nombre of ["unidades_edificio", "unidades_oportunidad"]) {
    const num = document.createElement("input");
    Object.assign(num, { type: "number", min: 1, step: 1, value: 1 });
    form.append(crearCampo(nombre, num));
  }
  MODELO.numericas.nombres.forEach((nombre, j) => {
    if (DERIVADAS.includes(nombre)) return;
    const control = document.createElement("input");
    if (CASILLAS.includes(nombre)) control.type = "checkbox";
    else Object.assign(control, { type: "number", step: "any", placeholder: `vacío = ${+MODELO.numericas.medianas[j].toFixed(2)}` });
    form.append(crearCampo(nombre, control));
  });
}

function leerEntrada(form) {
  const e = {};
  // OTRO no está entre las categorías, así que vectorizar() lo manda a la columna "infrecuente"
  for (const c of MODELO.categoricas) e[c.nombre] = form.elements[c.nombre].value;
  for (const nombre of MODELO.numericas.nombres) {
    const el = form.elements[nombre];
    if (!el) continue;
    e[nombre] = el.type === "checkbox" ? (el.checked ? 1 : 0) : el.value;
  }
  const f = new Date(form.elements.fecha_entrada.value + "T00:00:00");
  const mes = f.getMonth() + 1;
  e.mes_sen = Math.sin((2 * Math.PI * mes) / 12);
  e.mes_cos = Math.cos((2 * Math.PI * mes) / 12);
  e.dia_semana = (f.getDay() + 6) % 7;  // pandas: lunes = 0
  e.unidades_edificio_a_la_fecha = Math.log1p(Math.max(1, Number(form.elements.unidades_edificio.value) || 1));
  e.unidades_oportunidad_a_la_fecha = Math.log1p(Math.max(1, Number(form.elements.unidades_oportunidad.value) || 1));
  return e;
}

function mostrar(form) {
  const p = predecir(leerEntrada(form));
  document.getElementById("probabilidad").textContent = `${(100 * p).toFixed(1)} %`;
  document.getElementById("barra").style.width = `${100 * p}%`;
  const seguir = p >= MODELO.umbral_costo;
  const decision = document.getElementById("decision");
  decision.textContent = seguir ? "Seguimiento recomendado" : "Riesgo bajo de demora";
  decision.className = seguir ? "alerta" : "ok";
}

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("formulario");
  construirFormulario(form);
  document.getElementById("nombre-modelo").textContent = MODELO.nombre;
  document.getElementById("umbral").textContent = `${(100 * MODELO.umbral_costo).toFixed(1)} %`;
  form.addEventListener("input", () => mostrar(form));
  form.addEventListener("change", () => mostrar(form));
  mostrar(form);
});
