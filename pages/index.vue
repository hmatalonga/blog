<template>
  <section class="container">
    <h1 class="title">
      Hugo Matalonga
    </h1>
    <h2 class="subtitle">
      I code & make things.
    </h2>
    <ul class="links">
      <li><a href="https://github.com/hmatalonga">GitHub</a></li>
      <li><a href="https://www.linkedin.com/in/hmatalonga">LinkedIn</a></li>
      <li><a href="https://twitter.com/hmatalonga">Twitter</a></li>
      <li><a href="mailto:hello@hmatalonga.com">Email</a></li>
    </ul>
    <div class="copyright">
      <p>SVG background patterns by <a href="http://www.steveschoger.com/">@steveschoger</a></p>
    </div>
  </section>
</template>

<script>
export default {
}
</script>

<style>
.container {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  text-align: center;
}

.title {
  font-size: 84px;
  font-weight: 500;
  color: white;
  letter-spacing: -.1rem;
  margin-bottom: 20px;
}

.subtitle {
  font-size: 16px;
  font-weight: 400;
  text-transform: uppercase;
}

.links {
  position: fixed;
  width: 100%;
  bottom: 10vh;
  padding: 0;
  list-style: none;
}

.links li {
  display: inline;
  padding: 0 10px;
}

.links li:first-child {
  padding-left: 0;
}

.links li:last-child {
  padding-right: 0;
}

.links li > a {
  color: hsl(0, 0%, 84%);
  text-decoration: none;
  transition: color .15s ease-in-out;
}
.links li > a:hover {
  color: white;
}

.copyright {
  position: fixed;
  width: 100%;
  bottom: 2vh;
  padding: 0;
  font-size: 11px;
}

.copyright a {
  color: hsl(0, 0%, 84%);
  transition: color .15s ease-in-out;
}

.copyright a:hover {
  color: white;
}
</style>
