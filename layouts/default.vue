<template>
  <div
    class="flex flex-col font-sans max-w-screen-md pt-4 px-6 mx-auto md:pt-6 lg:px-0 mb-12 xl:pt-8"
  >
    <Header />
    <Nuxt />
  </div>
</template>

<script>
import Header from '~/components/Header.vue'
export default {
  components: {
    Header,
  },
}
</script>

<style>
.nuxt-link-exact-active {
  @apply text-gray-800;
}
</style>
