<template>
  <div
    class="flex flex-col items-center justify-center mt-10 pt-10 border-t border-gray-300"
  >
    <img
      class="w-16 h-16 rounded-full"
      src="~/assets/images/profile.jpg"
      alt="Hugo Matalonga"
    />
    <div class="mt-2 text-center">
      <div class="text-gray-500 text-sm">Written by</div>
      <div class="text-gray-800 text-lg font-semibold">Hugo Matalonga</div>
    </div>
    <div class="pb-12 mt-8">
      <nuxt-link
        class="text-indigo-500 font-semibold hover:text-indigo-600"
        to="/blog"
        >← View All Posts</nuxt-link
      >
    </div>
  </div>
</template>

<script>
export default {}
</script>

<style></style>
