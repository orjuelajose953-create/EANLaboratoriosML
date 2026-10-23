# ¿Esta orden tardará más de 60 días?

Página web del modelo del Proyecto Final Integrador de Machine Learning 1: estima la probabilidad de que la
puesta en servicio de una orden de conexión de gas natural tarde más de 60 días.

**Página publicada:** https://orjuelajose953-create.github.io/EANLaboratoriosML/proyectofinal-web/

* El modelo es el bosque aleatorio del proyecto (200 árboles), exportado a `modelo.js`.
* La predicción se calcula en el navegador, sin servidor, replicando el preprocesamiento de scikit-learn
  (imputación, one-hot y escalado) y el promedio de los árboles.
* Se recomienda seguimiento cuando la probabilidad supera el umbral por costo (31,8 %), elegido con un falso
  negativo tres veces más costoso que un falso positivo.

Los datos, el cuaderno y el informe están en un repositorio privado. Esta copia pública no incluye órdenes de
la base de datos: las firmas instaladoras aparecen solo con su seudónimo.

**Integrantes:** David Castelblanco, Fabian Sanchez, Alvaro Silva, José Orjuela
